"""Build a side-by-side comparison of ground truth, YOLO and Mask2Former.

Uses the COCO test split as ground truth. For each of a few test tiles it
renders three panels (GT, YOLO, Mask2Former) with class-colored masks and
saves them under reports/comparison_v5/.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

TEST_DIR = Path("data/processed/dataset_coco_v5/test")
GT_JSON = TEST_DIR / "_annotations.coco.json"
YOLO_WEIGHTS = Path("models/yolo_seg/weights/best.pt")
M2F_DIR = Path("models/mask2former")
OUT_DIR = Path("reports/comparison_v5")

N_SAMPLES = 4
CONF_YOLO = 0.25
CONF_M2F = 0.5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# fixed color per class name for consistency across panels
COLORS = {
    "building": (232, 65, 24),
    "trees_and_bushes": (34, 189, 90),
}


def overlay(image: Image.Image, mask_per_class: dict[str, np.ndarray]) -> Image.Image:
    """Blend a class -> HxW binary-mask dict onto a copy of the image."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for name, mask in mask_per_class.items():
        color = COLORS.get(name, (255, 255, 0)) + (110,)
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask > 0] = color
        overlay = Image.alpha_composite(overlay, Image.fromarray(rgba))
    return Image.alpha_composite(base, overlay).convert("RGB")


def label_image(img: Image.Image, text: str) -> Image.Image:
    """Draw a title bar on top of the image."""
    out = Image.new("RGB", (img.width, img.height + 40), "white")
    out.paste(img, (0, 40))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 6), text, fill="black", font=font)
    return out


def gt_masks(coco: dict, img_id: int, h: int, w: int, id2name: dict) -> dict[str, np.ndarray]:
    from PIL import ImageDraw as D
    per_class: dict[str, np.ndarray] = {}
    for a in coco["annotations"]:
        if a["image_id"] != img_id:
            continue
        name = id2name[a["category_id"]]
        if name not in COLORS:
            continue
        mask = per_class.setdefault(name, np.zeros((h, w), dtype=np.uint8))
        m_img = Image.fromarray(mask)
        for poly in a["segmentation"]:
            pts = [tuple(map(int, poly[i:i + 2])) for i in range(0, len(poly), 2)]
            D.Draw(m_img).polygon(pts, fill=1)
        per_class[name] = np.array(m_img)
    return per_class


def yolo_masks(model, img_path: Path, class_names: list[str]) -> dict[str, np.ndarray]:
    res = model.predict(str(img_path), conf=CONF_YOLO, device=DEVICE, verbose=False)[0]
    per_class: dict[str, np.ndarray] = {}
    if res.masks is None:
        return per_class
    masks = res.masks.data.cpu().numpy()   # (N, H, W) at native size
    cls = res.boxes.cls.cpu().numpy().astype(int)
    for m, c in zip(masks, cls):
        name = class_names[c]
        agg = per_class.setdefault(name, np.zeros(m.shape, dtype=np.uint8))
        agg |= (m > 0.5).astype(np.uint8)
    return per_class


def m2f_masks(model, processor, image: Image.Image, id2name: dict[int, str]) -> dict[str, np.ndarray]:
    with torch.no_grad():
        enc = processor(images=[image], return_tensors="pt")
        out = model(pixel_values=enc["pixel_values"].to(DEVICE))
    h, w = image.size[1], image.size[0]
    result = processor.post_process_instance_segmentation(
        out, target_sizes=[(h, w)], threshold=CONF_M2F
    )[0]
    seg = result["segmentation"].cpu().numpy()
    per_class: dict[str, np.ndarray] = {}
    for info in result["segments_info"]:
        name = id2name[info["label_id"]]
        m = (seg == info["id"]).astype(np.uint8)
        agg = per_class.setdefault(name, np.zeros_like(m))
        agg |= m
    return per_class


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coco = json.loads(GT_JSON.read_text())
    id2name = {c["id"]: c["name"] for c in coco["categories"]}

    yolo = YOLO(str(YOLO_WEIGHTS))
    yolo_names = [yolo.names[i] for i in range(len(yolo.names))]

    processor = Mask2FormerImageProcessor.from_pretrained(M2F_DIR)
    m2f = Mask2FormerForUniversalSegmentation.from_pretrained(M2F_DIR).to(DEVICE)
    m2f.eval()
    m2f_id2name = {int(k): v for k, v in m2f.config.id2label.items()}

    random.seed(0)
    sample = random.sample(coco["images"], min(N_SAMPLES, len(coco["images"])))

    for img_info in sample:
        img_path = TEST_DIR / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        h, w = image.size[1], image.size[0]

        gt_panel = label_image(overlay(image, gt_masks(coco, img_info["id"], h, w, id2name)), "ground truth")
        yolo_panel = label_image(overlay(image, yolo_masks(yolo, img_path, yolo_names)), "YOLOv8n-seg")
        m2f_panel = label_image(overlay(image, m2f_masks(m2f, processor, image, m2f_id2name)), "Mask2Former + Swin-T")

        strip = Image.new("RGB", (w * 3, gt_panel.height), "white")
        strip.paste(gt_panel, (0, 0))
        strip.paste(yolo_panel, (w, 0))
        strip.paste(m2f_panel, (w * 2, 0))
        out_name = img_info["file_name"].split(".rf.")[0] + ".jpg"
        strip.save(OUT_DIR / out_name, quality=90)
        print(f"wrote {out_name}")


if __name__ == "__main__":
    main()
