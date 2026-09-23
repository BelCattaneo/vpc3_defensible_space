"""Fine-tune Mask2Former with Swin-T backbone on the Defensible Space dataset."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

DATA_DIR = Path("data/processed/dataset_coco_v5")
MODEL_OUT = Path("models/mask2former")           # last epoch checkpoint
MODEL_BEST = Path("models/mask2former_best")     # lowest val_loss checkpoint
MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"

EPOCHS = 20
LR = 5e-5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

KEEP_CLASSES = ["building", "trees_and_bushes"]


class CocoSegDataset(Dataset):
    """Load a Roboflow COCO segmentation split for Mask2Former.

    If augment=True, applies random horizontal flip + color jitter
    (equivalent to a subset of ultralytics YOLO's defaults: fliplr=0.5,
    hsv_s=0.7, hsv_v=0.4, hsv_h=0.015). Geometric transforms are applied
    consistently to image and masks; color jitter only touches the image.
    """

    def __init__(self, split_dir: Path, keep_classes: list[str], augment: bool = False):
        coco = json.loads((split_dir / "_annotations.coco.json").read_text())
        self.split_dir = split_dir
        self.augment = augment
        self.color_jitter = ColorJitter(brightness=0.4, saturation=0.7, hue=0.015)

        # map coco category id -> contiguous label id (skip classes we do not keep)
        keep_cats = [c for c in coco["categories"] if c["name"] in keep_classes]
        keep_cats.sort(key=lambda c: c["name"])
        self.cat_to_label = {c["id"]: i for i, c in enumerate(keep_cats)}

        # group anns by image id
        self.anns_by_img = {}
        for a in coco["annotations"]:
            if a["category_id"] not in self.cat_to_label:
                continue
            self.anns_by_img.setdefault(a["image_id"], []).append(a)

        # only keep images that have at least one annotation
        self.images = [img for img in coco["images"] if img["id"] in self.anns_by_img]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img_path = self.split_dir / info["file_name"]
        image = Image.open(img_path).convert("RGB")
        w, h = info["width"], info["height"]

        class_labels = []
        masks = []
        for a in self.anns_by_img[info["id"]]:
            class_labels.append(self.cat_to_label[a["category_id"]])
            # segmentation is a list of polygons in absolute px
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in a["segmentation"]:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                _fill_polygon(mask, pts)
            masks.append(mask)

        masks_arr = np.stack(masks) if masks else np.zeros((0, h, w), dtype=np.uint8)

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if masks_arr.size > 0:
                    masks_arr = masks_arr[:, :, ::-1].copy()
            image = self.color_jitter(image)

        return {
            "image": image,
            "class_labels": class_labels,
            "masks": masks_arr,
        }


def _fill_polygon(mask: np.ndarray, pts: np.ndarray):
    """Fill polygon into mask in place (naive scanline, no external deps)."""
    from PIL import ImageDraw
    img = Image.fromarray(mask)
    ImageDraw.Draw(img).polygon([tuple(p) for p in pts], fill=1)
    mask[:] = np.array(img)


def collate(batch, processor):
    images = [b["image"] for b in batch]
    class_labels = [torch.tensor(b["class_labels"], dtype=torch.int64) for b in batch]
    mask_labels = [torch.tensor(b["masks"], dtype=torch.float32) for b in batch]

    enc = processor(images=images, return_tensors="pt")
    enc["class_labels"] = class_labels
    enc["mask_labels"] = mask_labels
    return enc


def main():
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    # cap image size to 384x384 to keep MPS memory in check
    processor = Mask2FormerImageProcessor.from_pretrained(
        MODEL_ID, size={"shortest_edge": 384, "longest_edge": 384}
    )
    # our class order matches the dataset (alphabetical over KEEP_CLASSES).
    id2label = {i: name for i, name in enumerate(sorted(KEEP_CLASSES))}
    label2id = {name: i for i, name in id2label.items()}
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_ID,
        num_labels=len(KEEP_CLASSES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    train_ds = CocoSegDataset(DATA_DIR / "train", KEEP_CLASSES, augment=True)
    val_ds = CocoSegDataset(DATA_DIR / "valid", KEEP_CLASSES, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=lambda b: collate(b, processor),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=lambda b: collate(b, processor),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(
                pixel_values=batch["pixel_values"].to(DEVICE),
                class_labels=[c.to(DEVICE) for c in batch["class_labels"]],
                mask_labels=[m.to(DEVICE) for m in batch["mask_labels"]],
            )
            loss = out.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del out, loss, batch
            if step % 20 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    pixel_values=batch["pixel_values"].to(DEVICE),
                    class_labels=[c.to(DEVICE) for c in batch["class_labels"]],
                    mask_labels=[m.to(DEVICE) for m in batch["mask_labels"]],
                )
                val_loss += out.loss.item()
                del out, batch

        val_avg = val_loss / len(val_loader)
        print(
            f"epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss / len(train_loader):.4f} "
            f"val_loss={val_avg:.4f}"
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if val_avg < best_val:
            best_val = val_avg
            model.save_pretrained(MODEL_BEST)
            processor.save_pretrained(MODEL_BEST)
            print(f"  new best (val={val_avg:.4f}) -> saved to {MODEL_BEST}")

    # always save the last-epoch checkpoint too (val_loss is noisy on tiny sets)
    model.save_pretrained(MODEL_OUT)
    processor.save_pretrained(MODEL_OUT)
    print(f"last epoch saved to {MODEL_OUT}, best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
