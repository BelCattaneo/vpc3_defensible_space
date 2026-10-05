"""Evaluate a fine-tuned YOLO segmentation model with pycocotools.

Runs inference on a COCO test split, encodes masks as RLE, and computes
mAP with the same ``pycocotools`` pipeline used by ``evaluate_m2f.py``.
Produces apples-to-apples numbers against Mask2Former on the same GT.

Overridable via env vars: ``YOLO_WEIGHTS`` (weights path), ``DATASET_COCO``
(dataset dir containing ``test/``), ``YOLO_REPORT`` (markdown output),
``YOLO_THRESHOLD`` (prediction confidence, default 0.001 to match
``ultralytics.val()``'s default).
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from constants import DEVICE
from models_io import load_yolo

WEIGHTS = Path(os.environ.get("YOLO_WEIGHTS", "models/yolo_seg/weights/best.pt"))
TEST_DIR = Path(os.environ.get("DATASET_COCO", "data/processed/dataset_coco_v6"), "test")
GT_PATH = TEST_DIR / "_annotations.coco.json"
REPORT = Path(os.environ.get("YOLO_REPORT", "reports/yolo_metrics.md"))

# Same conf threshold Ultralytics uses by default in val(), so a low score
# tail reaches COCO AP integration (as discussed with Mask2Former: the
# threshold is a filter, not a classification boundary for mAP).
YOLO_THRESHOLD = float(os.environ.get("YOLO_THRESHOLD", "0.001"))

MIN_POLYGON_POINTS = 6  # 3 (x, y) pairs

# Roboflow puts a container class covering the union of foreground classes.
# It appears in some COCO exports and distorts per-class mAP.
# Variantes conocidas del container class que Roboflow pone sobre la union
# de las clases reales. Se filtran del desglose per-class (COCO ya los
# excluye del promedio porque dan mAP=-1 si no hay anotaciones directas).
ROOT_CLASSES = {
    "defensible-space-v1", "defensible-space-v1-AJLD",
    "defensible-space-oracular",
}


def encode_mask(bin_mask: np.ndarray) -> dict:
    """Encode an HxW uint8 binary mask into COCO RLE."""
    rle = coco_mask.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _sanitize_gt(gt_path: Path) -> Path:
    """Drop annotations with empty/malformed segmentation; write a temp COCO json.

    Mirrors ``evaluate_m2f._sanitize_gt`` so that both evaluators measure
    against the same ground truth.
    """
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


def _predict(model, class_names: list[str], imgsz: int,
             gt_coco: COCO, name_to_gt_id: dict[str, int]) -> list[dict]:
    """Run YOLO over the test split and return COCO-format detections."""
    predictions: list[dict] = []
    for img_info in gt_coco.dataset["images"]:
        img_path = TEST_DIR / img_info["file_name"]
        with Image.open(img_path) as im:
            arr = np.array(im.convert("RGB"))
        h, w = arr.shape[:2]
        res = model.predict(arr, conf=YOLO_THRESHOLD, imgsz=imgsz,
                            device=DEVICE, verbose=False)[0]
        if res.masks is None:
            continue
        masks = res.masks.data.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy().astype(float)
        for mask, c, s in zip(masks, cls, scores):
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_LINEAR)
            bin_mask = (mask > 0.5).astype(np.uint8)
            if bin_mask.sum() == 0:
                continue
            class_name = class_names[c]
            if class_name not in name_to_gt_id:
                continue
            predictions.append({
                "image_id": img_info["id"],
                "category_id": name_to_gt_id[class_name],
                "segmentation": encode_mask(bin_mask),
                "score": float(s),
            })
    return predictions


def _write_report(gt_coco: COCO, pred_coco: COCO,
                  name_to_gt_id: dict[str, int]) -> None:
    """Compute overall + per-class mAP and write the markdown report."""
    ev = COCOeval(gt_coco, pred_coco, iouType="segm")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    lines = [
        "# YOLO metrics (test split, pycocotools)\n",
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

    model, class_names = load_yolo(WEIGHTS)
    imgsz = int(model.overrides.get("imgsz")
                or getattr(model, "args", {}).get("imgsz", 640))
    print(f"imgsz from checkpoint: {imgsz}")

    predictions = _predict(model, class_names, imgsz, gt_coco, name_to_gt_id)
    if not predictions:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("YOLO produced no predictions above threshold.\n")
        print("no predictions")
        return

    pred_coco = gt_coco.loadRes(predictions)
    _write_report(gt_coco, pred_coco, name_to_gt_id)


if __name__ == "__main__":
    main()
