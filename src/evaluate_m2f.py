"""Evaluate fine-tuned Mask2Former on the COCO test split.

Runs inference, converts predictions to COCO detection format (RLE masks),
then computes segmentation mAP with pycocotools against the same ground
truth used by YOLO. Results are written to reports/m2f_metrics.md.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

MODEL_DIR = Path("models/mask2former")
TEST_DIR = Path("data/processed/dataset_coco_v5/test")
GT_PATH = TEST_DIR / "_annotations.coco.json"
REPORT = Path("reports/m2f_metrics.md")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# our contiguous labels (as trained) mapped to the coco gt category ids.
# resolved dynamically below by name; see main().


def encode_mask(bin_mask: np.ndarray) -> dict:
    """Encode a HxW uint8 binary mask into COCO RLE."""
    rle = coco_mask.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _sanitize_gt(gt_path: Path) -> Path:
    """Drop annotations with empty/malformed segmentation, write a temp COCO json."""
    data = json.loads(gt_path.read_text())
    valid = []
    for a in data["annotations"]:
        s = a.get("segmentation")
        if not isinstance(s, list) or not s:
            continue
        if not all(isinstance(p, list) and len(p) >= 6 for p in s):
            continue
        valid.append(a)
    dropped = len(data["annotations"]) - len(valid)
    data["annotations"] = valid
    out = gt_path.with_name("_annotations.clean.json")
    out.write_text(json.dumps(data))
    print(f"dropped {dropped} annotations with invalid segmentation")
    return out


def main():
    clean_gt = _sanitize_gt(GT_PATH)
    gt_coco = COCO(str(clean_gt))
    name_to_gt_id = {c["name"]: c["id"] for c in gt_coco.dataset["categories"]}

    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_DIR)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    # trained label order comes from the model config (id2label)
    id2label = model.config.id2label
    label_to_gt_id = {int(k): name_to_gt_id[v] for k, v in id2label.items()}

    predictions = []
    for img_info in gt_coco.dataset["images"]:
        img_path = TEST_DIR / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        h, w = img_info["height"], img_info["width"]

        with torch.no_grad():
            enc = processor(images=[image], return_tensors="pt")
            out = model(pixel_values=enc["pixel_values"].to(DEVICE))

        result = processor.post_process_instance_segmentation(
            out, target_sizes=[(h, w)], threshold=0.5
        )[0]

        seg = result["segmentation"].cpu().numpy()   # HxW, int (query id or -1)
        for info in result["segments_info"]:
            q_id = info["id"]
            bin_mask = (seg == q_id).astype(np.uint8)
            if bin_mask.sum() == 0:
                continue
            predictions.append({
                "image_id": img_info["id"],
                "category_id": label_to_gt_id[info["label_id"]],
                "segmentation": encode_mask(bin_mask),
                "score": float(info["score"]),
            })

    if not predictions:
        REPORT.write_text("Mask2Former produced no predictions above threshold.\n")
        print("no predictions")
        return

    pred_coco = gt_coco.loadRes(predictions)
    ev = COCOeval(gt_coco, pred_coco, iouType="segm")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # per-class mAP too
    lines = ["# Mask2Former metrics (test split)\n"]
    lines.append(f"- overall mAP (0.5:0.95): **{ev.stats[0]:.3f}**\n")
    lines.append(f"- overall mAP@0.5: **{ev.stats[1]:.3f}**\n")
    lines.append("\n## Per-class mAP (0.5:0.95)\n")
    for name, gt_id in name_to_gt_id.items():
        if name == "defensible-space-v1":
            continue
        ev_c = COCOeval(gt_coco, pred_coco, iouType="segm")
        ev_c.params.catIds = [gt_id]
        ev_c.evaluate()
        ev_c.accumulate()
        ev_c.summarize()
        lines.append(f"- {name}: mAP={ev_c.stats[0]:.3f}, mAP@0.5={ev_c.stats[1]:.3f}\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
