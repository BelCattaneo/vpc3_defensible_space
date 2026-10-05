"""Small mask / raster helpers shared across scripts."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw


def fill_polygon(mask: np.ndarray, polygon: Sequence[float]) -> None:
    """Fill a COCO-style polygon [x1, y1, x2, y2, ...] into a HxW uint8 mask in place.

    Coordinates are int-truncated before rasterization to match the previous
    inline implementations in train_mask2former.py and compare_models.py.
    """
    pts = [(int(polygon[i]), int(polygon[i + 1])) for i in range(0, len(polygon), 2)]
    img = Image.fromarray(mask)
    ImageDraw.Draw(img).polygon(pts, fill=1)
    mask[:] = np.array(img)


def read_tif_as_rgb(path: Path) -> np.ndarray:
    """Read a TIF via rasterio and return an HxWx3 uint8 RGB array (alpha dropped)."""
    with rasterio.open(path) as src:
        data = src.read()  # (bands, H, W)
    img = np.transpose(data, (1, 2, 0))
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img
