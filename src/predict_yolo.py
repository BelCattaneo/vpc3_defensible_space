"""Run YOLO segmentation on test tiles and save visual overlays."""

from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path("models/yolo_seg/weights/best.pt")
TEST_DIR = Path("data/processed/dataset_yolo_v5/test/images")
OUT_DIR = Path("reports/yolo_predictions")

model = YOLO(str(WEIGHTS))

model.predict(
    source=str(TEST_DIR),
    imgsz=1024,
    conf=0.25,          # keep low so we see confusions, not just strong hits
    device="mps",
    save=True,          # writes images with masks and class labels
    project=str(OUT_DIR.parent),
    name=OUT_DIR.name,
    exist_ok=True,
)
