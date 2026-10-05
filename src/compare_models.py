"""Build a side-by-side comparison of ground truth, YOLO and Mask2Former.

Uses the COCO test split as ground truth. For each of a few test tiles it
renders three panels (GT, YOLO, Mask2Former) with class-colored masks and
saves them under ``reports/comparison_v6/``.
"""

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from constants import DEVICE
from mask_utils import fill_polygon
from models_io import load_mask2former, load_yolo, m2f_instance_masks

TEST_DIR = Path("data/processed/dataset_coco_v6/test")
GT_JSON = TEST_DIR / "_annotations.coco.json"
YOLO_WEIGHTS = Path(os.environ.get("YOLO_WEIGHTS", "models/yolo_seg/weights/best.pt"))
M2F_DIR = Path(os.environ.get("M2F_MODEL_DIR", "models/mask2former"))
YOLO_LABEL = os.environ.get("YOLO_LABEL", "YOLOv8n-seg")
M2F_LABEL = os.environ.get("M2F_LABEL", "Mask2Former + Swin-T")
OUT_DIR = Path(os.environ.get("OUT_DIR", "reports/comparison_v6"))

N_SAMPLES = 4
# Mismo threshold de confianza para los dos modelos. Comparar M2F a 0,5
# contra YOLO a 0,25 produce figuras no comparables (YOLO muestra mas
# cantidad de detecciones de menor confianza que M2F simplemente deja
# fuera). 0,5 es el punto operativo estandar documentado por HF para
# ``post_process_instance_segmentation``.
CONF_SAME = float(os.environ.get("COMPARE_CONF", "0.5"))
CONF_YOLO = CONF_SAME
CONF_M2F = CONF_SAME

TITLE_BAR_HEIGHT = 40
TITLE_FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"
TITLE_FONT_SIZE = 24
OVERLAY_ALPHA = 110  # 0-255; used for class-mask overlays

# Fixed color per class name for consistency across panels.
COLORS: dict[str, tuple[int, int, int]] = {
    "building": (232, 65, 24),
    "trees_and_bushes": (34, 189, 90),
}
FALLBACK_COLOR: tuple[int, int, int] = (255, 255, 0)


def overlay(image: Image.Image, mask_per_class: dict[str, np.ndarray]) -> Image.Image:
    """Blend a ``{class_name: HxW binary mask}`` dict onto a copy of the image."""
    base = image.convert("RGBA")
    combined = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for name, mask in mask_per_class.items():
        color = COLORS.get(name, FALLBACK_COLOR) + (OVERLAY_ALPHA,)
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask > 0] = color
        combined = Image.alpha_composite(combined, Image.fromarray(rgba))
    return Image.alpha_composite(base, combined).convert("RGB")


def label_image(img: Image.Image, text: str) -> Image.Image:
    """Draw a white title bar with ``text`` on top of ``img``."""
    out = Image.new("RGB", (img.width, img.height + TITLE_BAR_HEIGHT), "white")
    out.paste(img, (0, TITLE_BAR_HEIGHT))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(TITLE_FONT_PATH, TITLE_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 6), text, fill="black", font=font)
    return out


def gt_masks(
    coco: dict, img_id: int, h: int, w: int, id2name: dict[int, str]
) -> dict[str, np.ndarray]:
    """Rasterize ground-truth polygons for one image into per-class masks."""
    per_class: dict[str, np.ndarray] = {}
    for a in coco["annotations"]:
        if a["image_id"] != img_id:
            continue
        name = id2name[a["category_id"]]
        if name not in COLORS:
            continue
        mask = per_class.setdefault(name, np.zeros((h, w), dtype=np.uint8))
        for poly in a["segmentation"]:
            fill_polygon(mask, poly)
    return per_class


def yolo_masks(model, img_path: Path, class_names: list[str]) -> dict[str, np.ndarray]:
    """Run YOLO on one tile and return per-class aggregated binary masks.

    Lee ``imgsz`` del checkpoint para no inflar las mascaras: inferir a
    1024 contra un modelo entrenado a 640 produce blobs dilatados
    (mismo bug corregido en ``compute_alerts.predict_yolo``).
    """
    with Image.open(img_path) as im:
        w, h = im.size
    imgsz = int(model.overrides.get("imgsz")
                or getattr(model, "args", {}).get("imgsz", 640))
    res = model.predict(str(img_path), conf=CONF_YOLO, imgsz=imgsz,
                        device=DEVICE, verbose=False)[0]
    per_class: dict[str, np.ndarray] = {}
    if res.masks is None:
        return per_class
    masks = res.masks.data.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    for m, c in zip(masks, cls):
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)
        name = class_names[c]
        agg = per_class.setdefault(name, np.zeros((h, w), dtype=np.uint8))
        agg |= (m > 0.5).astype(np.uint8)
    return per_class


def m2f_masks(
    model, processor, image: Image.Image, id2name: dict[int, str]
) -> dict[str, np.ndarray]:
    """Run Mask2Former on one image and aggregate per-instance masks by class name."""
    target_hw = (image.size[1], image.size[0])
    per_class: dict[str, np.ndarray] = {}
    for label_id, bin_mask, _score in m2f_instance_masks(
        model, processor, image, target_hw, CONF_M2F
    ):
        name = id2name[label_id]
        agg = per_class.setdefault(name, np.zeros_like(bin_mask))
        agg |= bin_mask
    return per_class


def _compose_strip(
    image: Image.Image,
    gt_per_class: dict[str, np.ndarray],
    yolo_per_class: dict[str, np.ndarray],
    m2f_per_class: dict[str, np.ndarray],
) -> Image.Image:
    """Assemble the three labeled panels for one tile into a single strip."""
    gt_panel = label_image(overlay(image, gt_per_class), "ground truth")
    yolo_panel = label_image(overlay(image, yolo_per_class), YOLO_LABEL)
    m2f_panel = label_image(overlay(image, m2f_per_class), M2F_LABEL)

    w = image.width
    strip = Image.new("RGB", (w * 3, gt_panel.height), "white")
    strip.paste(gt_panel, (0, 0))
    strip.paste(yolo_panel, (w, 0))
    strip.paste(m2f_panel, (w * 2, 0))
    return strip


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coco = json.loads(GT_JSON.read_text())
    id2name = {c["id"]: c["name"] for c in coco["categories"]}

    yolo, yolo_names = load_yolo(YOLO_WEIGHTS)
    processor, m2f = load_mask2former(M2F_DIR)
    m2f_id2name = {int(k): v for k, v in m2f.config.id2label.items()}

    random.seed(0)
    sample = random.sample(coco["images"], min(N_SAMPLES, len(coco["images"])))

    for img_info in sample:
        img_path = TEST_DIR / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        h, w = image.size[1], image.size[0]

        strip = _compose_strip(
            image,
            gt_masks(coco, img_info["id"], h, w, id2name),
            yolo_masks(yolo, img_path, yolo_names),
            m2f_masks(m2f, processor, image, m2f_id2name),
        )
        out_name = img_info["file_name"].split(".rf.")[0] + ".jpg"
        strip.save(OUT_DIR / out_name, quality=90)
        print(f"wrote {out_name}")


if __name__ == "__main__":
    main()
