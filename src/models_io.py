"""Model loading helpers used across evaluation and comparison scripts."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)
from ultralytics import YOLO

from constants import DEVICE


def load_mask2former(
    model_dir: Path,
) -> tuple[Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation]:
    """Load a fine-tuned Mask2Former checkpoint in eval mode on DEVICE."""
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    return processor, model


def load_yolo(weights: Path) -> tuple[YOLO, list[str]]:
    """Load YOLO weights and return the model with its ordered class-names list."""
    model = YOLO(str(weights))
    names = [model.names[i] for i in range(len(model.names))]
    return model, names


def m2f_instance_masks(
    model: Mask2FormerForUniversalSegmentation,
    processor: Mask2FormerImageProcessor,
    image: Image.Image,
    target_hw: tuple[int, int],
    threshold: float,
) -> Iterator[tuple[int, np.ndarray, float]]:
    """Run Mask2Former on a PIL image and yield (label_id, binary_mask, score).

    ``target_hw`` is the (height, width) the mask should be resized to; passing
    the ground-truth size (from COCO metadata) keeps evaluation aligned, while
    passing ``(image.size[1], image.size[0])`` matches the visible image.
    """
    with torch.no_grad():
        enc = processor(images=[image], return_tensors="pt")
        out = model(pixel_values=enc["pixel_values"].to(DEVICE))

    result = processor.post_process_instance_segmentation(
        out, target_sizes=[target_hw], threshold=threshold
    )[0]
    seg = result["segmentation"].cpu().numpy()
    for info in result["segments_info"]:
        bin_mask = (seg == info["id"]).astype(np.uint8)
        yield int(info["label_id"]), bin_mask, float(info["score"])
