"""Evaluate fine-tuned Mask2Former on the COCO test split.

Runs inference, converts predictions to COCO detection format (RLE masks),
then computes segmentation mAP with pycocotools against the same ground
truth used by YOLO. Results are written to ``reports/m2f_metrics.md``.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models_io import load_mask2former, m2f_instance_masks

MODEL_DIR = Path(os.environ.get("M2F_MODEL_DIR", "models/mask2former"))
TEST_DIR = Path(os.environ.get("DATASET_COCO", "data/processed/dataset_coco_v6"), "test")
GT_PATH = TEST_DIR / "_annotations.coco.json"
REPORT = Path(os.environ.get("M2F_REPORT", "reports/m2f_metrics.md"))

# Roboflow adds a "root" class covering the union of foreground classes.
# It distorts per-class mAP so we skip it in the per-class breakdown.
# El set tiene las variantes conocidas de ambos exports (v6 y oracular_merged)
# para espejar la logica de ``evaluate_yolo``.
ROOT_CLASSES = {
    "defensible-space-v1", "defensible-space-v1-AJLD",
    "defensible-space-oracular",
}

M2F_THRESHOLD = float(os.environ.get("M2F_THRESHOLD", "0.5"))
MIN_POLYGON_POINTS = 6  # 3 (x, y) pairs


def encode_mask(bin_mask: np.ndarray) -> dict:
    """Encode an HxW uint8 binary mask into COCO RLE."""
    rle = coco_mask.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _sanitize_gt(gt_path: Path) -> Path:
    """Drop annotations with empty/malformed segmentation; write a temp COCO json."""
    data = json.loads(gt_path.read_text())
    valid = []
    for a in data["annotations"]:
        s = a.get("segmentation")
        if not isinstance(s, list) or not s:
            continue
        if not all(isinstance(p, list) and len(p) >= MIN_POLYGON_POINTS for p in s):
            continue
        valid.append(a)
    dropped = len(data["annotations"]) - len(valid)
    data["annotations"] = valid
    out = gt_path.with_name("_annotations.clean.json")
    out.write_text(json.dumps(data))
    print(f"dropped {dropped} annotations with invalid segmentation")
    return out


def _predict(model, processor, gt_coco: COCO, label_to_gt_id: dict[int, int]) -> list[dict]:
    """Run Mask2Former over the test split and return COCO-format detections."""
    predictions: list[dict] = []
    for img_info in gt_coco.dataset["images"]:
        img_path = TEST_DIR / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        target_hw = (img_info["height"], img_info["width"])

        for label_id, bin_mask, score in m2f_instance_masks(
            model, processor, image, target_hw, M2F_THRESHOLD
        ):
            if bin_mask.sum() == 0:
                continue
            predictions.append({
                "image_id": img_info["id"],
                "category_id": label_to_gt_id[label_id],
                "segmentation": encode_mask(bin_mask),
                "score": score,
            })
    return predictions


def _write_report(gt_coco: COCO, pred_coco: COCO, name_to_gt_id: dict[str, int]) -> None:
    """Compute overall + per-class mAP and write the markdown report."""
    ev = COCOeval(gt_coco, pred_coco, iouType="segm")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    lines = [
        "# Mask2Former metrics (test split)\n",
        f"- overall mAP (0.5:0.95): **{ev.stats[0]:.3f}**\n",
        f"- overall mAP@0.5: **{ev.stats[1]:.3f}**\n",
        "\n## Per-class mAP (0.5:0.95)\n",
    ]
    for name, gt_id in name_to_gt_id.items():
        if name in ROOT_CLASSES:
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


def main() -> None:
    clean_gt = _sanitize_gt(GT_PATH)
    gt_coco = COCO(str(clean_gt))
    name_to_gt_id = {c["name"]: c["id"] for c in gt_coco.dataset["categories"]}

    processor, model = load_mask2former(MODEL_DIR)
    # Trained label order comes from the model config (id2label).
    label_to_gt_id = {int(k): name_to_gt_id[v] for k, v in model.config.id2label.items()}

    predictions = _predict(model, processor, gt_coco, label_to_gt_id)
    if not predictions:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("Mask2Former produced no predictions above threshold.\n")
        print("no predictions")
        return

    pred_coco = gt_coco.loadRes(predictions)
    _write_report(gt_coco, pred_coco, name_to_gt_id)


if __name__ == "__main__":
    main()
