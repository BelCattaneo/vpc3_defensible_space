"""Fine-tune YOLOv8n-seg on the Defensible Space dataset."""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("data/processed/dataset_yolo_v5/data.yaml").absolute()
MODEL_OUT = Path("models/yolo_seg").absolute()   # absolute path avoids Ultralytics prepending runs/segment/ from cwd

model = YOLO("yolov8n-seg.pt")   # pretrained on COCO

model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=1024,
    batch=4,
    device="mps",       # Apple Silicon GPU
    project=str(MODEL_OUT.parent),
    name=MODEL_OUT.name,
    exist_ok=True,
    seed=42,
)
