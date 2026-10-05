# Experimentos: comandos de reproducibilidad

Un comando por run. Las métricas finales y los archivos de pesos se registran
en `reports/HISTORY.md`. Los pesos quedan archivados en `models/_archive/<run>/`.

Todos los scripts de training aceptan los parámetros vía variables de entorno:

- `DATASET_YOLO` o `DATASET_COCO`: directorio del dataset.
- `YOLO_MODEL`: checkpoint pre-entrenado (`yolov8n-seg.pt`, `yolov8l-seg.pt`, etc.).
- `YOLO_PATIENCE`: `patience` para early stop (default `100` = efectivamente sin ES).
- `YOLO_BATCH`: batch size (default `4`).
- `YOLO_LR0`: learning rate inicial (default `0` = heurística de Ultralytics).
- `YOLO_OPTIMIZER`: `auto`, `AdamW`, `SGD`, etc. (default `auto`).

Para Mask2Former las augmentations se cambian editando `AUGMENT` en `src/train_mask2former.py`.

## YOLOv8n-seg

### v5 (86 autora, baseline sin early stop)

```bash
DATASET_YOLO=data/processed/dataset_yolo_v5 \
YOLO_MODEL=yolov8n-seg.pt \
YOLO_PATIENCE=100 \
  uv run python -u src/train_yolo_seg.py
```

### v6 (927 mix, aug extra)

Requiere editar el script para agregar `flipud=0.5, degrees=10, copy_paste=0.3`
en los kwargs de `model.train`. El resto:

```bash
DATASET_YOLO=data/processed/dataset_yolo_v6 \
YOLO_MODEL=yolov8n-seg.pt \
YOLO_PATIENCE=100 \
  uv run python -u src/train_yolo_seg.py
```

### v8 (927 mix, solo defaults)

```bash
DATASET_YOLO=data/processed/dataset_yolo_v6 \
YOLO_MODEL=yolov8n-seg.pt \
YOLO_PATIENCE=100 \
  uv run python -u src/train_yolo_seg.py
```

### v9 (86 autora, early stop)

```bash
DATASET_YOLO=data/processed/dataset_yolo_v5 \
YOLO_MODEL=yolov8n-seg.pt \
YOLO_PATIENCE=5 \
  uv run python -u src/train_yolo_seg.py
```

### v10 (486 autora, early stop)

```bash
DATASET_YOLO=data/processed/dataset_yolo_v7 \
YOLO_MODEL=yolov8n-seg.pt \
YOLO_PATIENCE=5 \
  uv run python -u src/train_yolo_seg.py
```

## YOLOv8l-seg

Modelo grande (46 M parámetros). El `optimizer=auto` elige una tasa de aprendizaje
que colapsa el entrenamiento con 86 imágenes, por lo que se fija explícitamente.

### v11 (86 autora)

```bash
DATASET_YOLO=data/processed/dataset_yolo_v5 \
YOLO_MODEL=yolov8l-seg.pt \
YOLO_BATCH=2 \
YOLO_PATIENCE=100 \
YOLO_OPTIMIZER=AdamW \
YOLO_LR0=0.0005 \
  uv run python -u src/train_yolo_seg.py
```

### v12 (927 mix)

Con 927 imágenes y `imgsz=1024` el modelo satura los 24 GB del M4 (OOM en epoch 7
en una primera corrida). Se bajó a `imgsz=640` y `epochs=30` para que entre en
memoria y termine en tiempo razonable. La reducción de resolución es una
limitación de hardware que vale documentar en el informe.

```bash
DATASET_YOLO=data/processed/dataset_yolo_v6 \
YOLO_MODEL=yolov8l-seg.pt \
YOLO_BATCH=2 \
YOLO_WORKERS=0 \
YOLO_PATIENCE=100 \
YOLO_EPOCHS=30 \
YOLO_IMG_SIZE=640 \
YOLO_OPTIMIZER=AdamW \
YOLO_LR0=0.0005 \
  uv run python -u src/train_yolo_seg.py
```

## Mask2Former con backbone Swin-T

Config fija en el script: `IMAGE_SIZE=384`, `LR=5e-5`, `batch=1` (por memoria MPS).

### v4 (86 autora, 20 epochs sin early stop)

Requiere `EPOCHS=20` y `EARLY_STOP_PATIENCE=100` en el script, y la pipeline
mínima de augmentation (`hflip` + `ColorJitter`).

```bash
DATASET_COCO=data/processed/dataset_coco_v5 \
  uv run python -u src/train_mask2former.py
```

### v7 (927 mix, Albumentations completo)

Requiere cambiar `AUGMENT` en el script a:

```python
AUGMENT = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), p=0.5),
    A.ColorJitter(brightness=0.4, saturation=0.7, hue=0.015, p=0.5),
])
```

```bash
DATASET_COCO=data/processed/dataset_coco_v6 \
  uv run python -u src/train_mask2former.py
```

### v8 (927 mix, aug mínima)

Pipeline mínima en `AUGMENT` (`hflip` + `ColorJitter`), que es la por defecto en
el script.

```bash
DATASET_COCO=data/processed/dataset_coco_v6 \
  uv run python -u src/train_mask2former.py
```

### v9 (86 autora, early stop)

Igual que v8 pero sobre `dataset_coco_v5` y con `EARLY_STOP_PATIENCE=5` en el script.

```bash
DATASET_COCO=data/processed/dataset_coco_v5 \
  uv run python -u src/train_mask2former.py
```

### v10 (486 autora)

```bash
DATASET_COCO=data/processed/dataset_coco_v7 \
  uv run python -u src/train_mask2former.py
```

## Evaluación cruzada sobre held-out team

Construir el held-out:

```bash
uv run python -u src/build_holdout_team.py
uv run python -u src/coco_to_yolo.py \
  data/processed/dataset_coco_holdout_team \
  data/processed/dataset_yolo_holdout_team
```

Evaluar un checkpoint YOLO:

```python
from ultralytics import YOLO
m = YOLO("models/_archive/yolo_v8/weights/best.pt")
m.val(data="data/processed/dataset_yolo_holdout_team/data.yaml",
      split="test", device="mps")
```

Evaluar un checkpoint M2F:

```bash
M2F_MODEL_DIR=models/_archive/m2f_v8/best \
DATASET_COCO=data/processed/dataset_coco_holdout_team \
  uv run python -u src/evaluate_m2f.py
```

## DeepForest zero-shot sobre held-out team

```bash
uv run python -u src/eval_deepforest.py
```
