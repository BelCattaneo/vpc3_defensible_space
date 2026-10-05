"""Fine-tune a YOLO segmentation model on the Defensible Space dataset.

Overridable via env vars: ``DATASET_YOLO`` (dataset dir), ``YOLO_MODEL``
(base checkpoint name, e.g. ``yolov8n-seg.pt`` or ``yolov8l-seg.pt``),
``YOLO_BATCH`` (batch size, defaults to 4), ``YOLO_PATIENCE`` (early-stop
patience, defaults to 100 = effectively disabled).
"""

import os
from pathlib import Path

from ultralytics import YOLO

from constants import DEVICE

_DEFAULT_DATASET = "data/processed/dataset_yolo_v6"
DATA_YAML = Path(os.environ.get("DATASET_YOLO", _DEFAULT_DATASET), "data.yaml").absolute()
MODEL_OUT = Path(os.environ.get("MODEL_OUT", "models/yolo_seg")).absolute()

BASE_MODEL = os.environ.get("YOLO_MODEL", "yolov8n-seg.pt")
EPOCHS = int(os.environ.get("YOLO_EPOCHS", "50"))
IMG_SIZE = int(os.environ.get("YOLO_IMG_SIZE", "1024"))
BATCH = int(os.environ.get("YOLO_BATCH", "4"))
SEED = 42
PATIENCE = int(os.environ.get("YOLO_PATIENCE", "100"))
# ``optimizer=auto`` picks lr0 by heuristic. For larger backbones (l/x)
# that lr can be too high and collapse training; override explicitly.
LR0 = float(os.environ.get("YOLO_LR0", "0"))  # 0 means "use ultralytics auto"
OPTIMIZER = os.environ.get("YOLO_OPTIMIZER", "auto")

def main() -> None:
    # Augmentation left at Ultralytics defaults (fliplr, hsv_h, hsv_s, hsv_v,
    # mosaic, translate, scale). An earlier attempt to add aerial-specific
    # augmentation (flipud, degrees, copy_paste) hurt the test metrics when
    # trained on the merged 927-image dataset, see reports/HISTORY.md.
    model = YOLO(BASE_MODEL)
    kw = {
        "data": str(DATA_YAML),
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": BATCH,
        "device": DEVICE,
        "project": str(MODEL_OUT.parent),
        "name": MODEL_OUT.name,
        "exist_ok": True,
        "seed": SEED,
        "patience": PATIENCE,
        "optimizer": OPTIMIZER,
    }
    if LR0 > 0:
        kw["lr0"] = LR0
    if "YOLO_WORKERS" in os.environ:
        kw["workers"] = int(os.environ["YOLO_WORKERS"])
    model.train(**kw)


if __name__ == "__main__":
    main()
