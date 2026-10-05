"""Shared constants across the pipeline scripts."""

import os

import torch

# Preferred torch device: honra DEVICE del entorno, si no autodetecta en orden
# NVIDIA CUDA, Apple Silicon MPS, CPU. Permite correr el pipeline en cualquier
# host sin pasar argumentos (NVIDIA Linux/Windows, Mac M-series, cualquier CPU).
_env_device = os.environ.get("DEVICE")
if _env_device:
    DEVICE = _env_device
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Foreground classes kept from the Roboflow COCO export.
# Order is alphabetical, which is the same order used to assign contiguous
# label ids at training time (see coco_to_yolo.py, train_mask2former.py).
KEEP_CLASSES: tuple[str, ...] = ("building", "trees_and_bushes")
