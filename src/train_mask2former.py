"""Fine-tune Mask2Former with Swin-T backbone on the Defensible Space dataset."""

import json
import os
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

from constants import DEVICE, KEEP_CLASSES
from mask_utils import fill_polygon

DATA_DIR = Path(os.environ.get("DATASET_COCO", "data/processed/dataset_coco_v6"))
MODEL_OUT = Path(os.environ.get("MODEL_OUT", "models/mask2former"))           # last epoch checkpoint
# ``_best`` checkpoint se deriva de ``MODEL_OUT`` para que dos runs
# simultáneos (p.ej. Swin-T y Swin-S) no se pisen el mejor entre sí.
MODEL_BEST = Path(os.environ.get(
    "MODEL_BEST",
    f"{MODEL_OUT}_best" if str(MODEL_OUT) != "models/mask2former"
    else "models/mask2former_best",
))

SEED = int(os.environ.get("M2F_SEED", "42"))
MODEL_ID = os.environ.get("M2F_MODEL_ID", "facebook/mask2former-swin-tiny-coco-instance")

EPOCHS = int(os.environ.get("M2F_EPOCHS", "20"))
LR = float(os.environ.get("M2F_LR", "5e-5"))

# Early stopping: stop if val_loss does not improve for this many epochs.
# Default 100 effectively disables ES (consistent with v4 baseline).
EARLY_STOP_PATIENCE = int(os.environ.get("M2F_EARLY_STOP_PATIENCE", "100"))

# Image size cap to keep MPS memory in check.
IMAGE_SIZE = 384

# Minimal augmentation pipeline (horizontal flip + color jitter), which
# produced the best test metrics with 86 clean labels (see v4 in HISTORY.md).
# The more aggressive aerial pipeline (vflip, rotate, random resized crop)
# was tried in v7 and hurt test mAP when the training set included team
# labels with inconsistent quality.
AUGMENT = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, saturation=0.7, hue=0.015, p=0.5),
], seed=SEED)

# Free MPS memory every N training steps.
MPS_EMPTY_CACHE_EVERY = 20


class CocoSegDataset(Dataset):
    """Load a Roboflow COCO segmentation split for Mask2Former.

    If ``augment=True``, applies random horizontal flip + color jitter
    (equivalent to a subset of ultralytics YOLO's defaults: fliplr=0.5,
    hsv_s=0.7, hsv_v=0.4, hsv_h=0.015). Geometric transforms are applied
    consistently to image and masks; color jitter only touches the image.
    """

    def __init__(self, split_dir: Path, keep_classes, augment: bool = False):
        coco = json.loads((split_dir / "_annotations.coco.json").read_text())
        self.split_dir = split_dir
        self.augment = augment

        # Map coco category id -> contiguous label id (skip classes we do not keep).
        keep_cats = [c for c in coco["categories"] if c["name"] in keep_classes]
        keep_cats.sort(key=lambda c: c["name"])
        self.cat_to_label = {c["id"]: i for i, c in enumerate(keep_cats)}

        # Group anns by image id.
        self.anns_by_img: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            if a["category_id"] not in self.cat_to_label:
                continue
            self.anns_by_img.setdefault(a["image_id"], []).append(a)

        # Only keep images that have at least one annotation.
        self.images = [img for img in coco["images"] if img["id"] in self.anns_by_img]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img_path = self.split_dir / info["file_name"]
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = info["height"], info["width"]

        class_labels: list[int] = []
        masks: list[np.ndarray] = []
        for a in self.anns_by_img[info["id"]]:
            class_labels.append(self.cat_to_label[a["category_id"]])
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in a["segmentation"]:
                fill_polygon(mask, poly)
            masks.append(mask)

        if self.augment:
            out = AUGMENT(image=image, masks=masks) if masks else AUGMENT(image=image)
            image = out["image"]
            masks = out.get("masks", masks)
            # Drop instances whose mask was fully cropped out by the geometric
            # transforms (otherwise Mask2Former gets an all-zero mask and the
            # Hungarian matcher misbehaves).
            kept = [(c, m) for c, m in zip(class_labels, masks) if m.sum() > 0]
            class_labels = [c for c, _ in kept]
            masks = [m for _, m in kept]

        masks_arr = (np.stack(masks) if masks
                     else np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8))
        return {
            "image": Image.fromarray(image),
            "class_labels": class_labels,
            "masks": masks_arr,
        }


def collate(batch, processor):
    """Turn a list of dataset items into a Mask2Former-ready batch."""
    images = [b["image"] for b in batch]
    class_labels = [torch.tensor(b["class_labels"], dtype=torch.int64) for b in batch]
    mask_labels = [torch.tensor(b["masks"], dtype=torch.float32) for b in batch]

    enc = processor(images=images, return_tensors="pt")
    enc["class_labels"] = class_labels
    enc["mask_labels"] = mask_labels
    return enc


def _forward(model, batch) -> torch.Tensor:
    """Forward pass on DEVICE, returning the loss tensor."""
    out = model(
        pixel_values=batch["pixel_values"].to(DEVICE),
        class_labels=[c.to(DEVICE) for c in batch["class_labels"]],
        mask_labels=[m.to(DEVICE) for m in batch["mask_labels"]],
    )
    return out.loss


def _train_one_epoch(model, loader, optimizer) -> float:
    """Run one training epoch, return average loss."""
    model.train()
    total = 0.0
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = _forward(model, batch)
        loss.backward()
        optimizer.step()
        total += loss.item()
        del loss, batch
        if step % MPS_EMPTY_CACHE_EVERY == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return total / len(loader)


def _validate(model, loader) -> float:
    """Run one validation epoch, return average loss."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            total += _forward(model, batch).item()
            del batch
    return total / len(loader)


def _set_seed(seed: int) -> None:
    """Siembra todos los generadores que afectan al training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    _set_seed(SEED)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    processor = Mask2FormerImageProcessor.from_pretrained(
        MODEL_ID, size={"shortest_edge": IMAGE_SIZE, "longest_edge": IMAGE_SIZE}
    )
    # Our class order matches the dataset (alphabetical over KEEP_CLASSES).
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
    epochs_since_improvement = 0
    for epoch in range(1, EPOCHS + 1):
        train_avg = _train_one_epoch(model, train_loader, optimizer)
        val_avg = _validate(model, val_loader)

        print(
            f"epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_avg:.4f} "
            f"val_loss={val_avg:.4f}"
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if val_avg < best_val:
            best_val = val_avg
            epochs_since_improvement = 0
            model.save_pretrained(MODEL_BEST)
            processor.save_pretrained(MODEL_BEST)
            print(f"  new best (val={val_avg:.4f}) -> saved to {MODEL_BEST}")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= EARLY_STOP_PATIENCE:
                print(f"  early stop: no val improvement in "
                      f"{EARLY_STOP_PATIENCE} epochs")
                break

    # Always save the last-epoch checkpoint too (val_loss is noisy on tiny sets).
    model.save_pretrained(MODEL_OUT)
    processor.save_pretrained(MODEL_OUT)
    print(f"last epoch saved to {MODEL_OUT}, best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
