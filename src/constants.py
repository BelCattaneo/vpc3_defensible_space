"""Shared constants across the pipeline scripts."""

import os

import torch

# Preferred torch device: Apple Silicon GPU when available, CPU otherwise.
_env_device = os.environ.get("DEVICE")
DEVICE = _env_device or ("mps" if torch.backends.mps.is_available() else "cpu")

# Foreground classes kept from the Roboflow COCO export.
# Order is alphabetical, which is the same order used to assign contiguous
# label ids at training time (see coco_to_yolo.py, train_mask2former.py).
KEEP_CLASSES: tuple[str, ...] = ("building", "trees_and_bushes")
