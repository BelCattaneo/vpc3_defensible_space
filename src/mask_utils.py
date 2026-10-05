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
    """Read a TIF via rasterio and return an HxWx3 uint8 RGB array (alpha dropped).

    GeoTIFF puede venir con dtype uint8/uint16/float32; se escala al rango
    [0, 255] y se convierte a uint8 para que ``Pillow`` lo acepte sin
    quejarse. Si no hay senal (todo cero) se devuelve igualmente uint8.
    """
    with rasterio.open(path) as src:
        data = src.read()  # (bands, H, W)
    img = np.transpose(data, (1, 2, 0))
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        maxv = img.max() if img.size else 1
        if maxv > 0:
            img = np.clip(img / maxv * 255.0, 0, 255)
        img = img.astype(np.uint8)
    return img
