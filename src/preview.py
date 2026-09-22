"""Generate PNG previews of N random tiles for visual inspection."""

import random
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

INPUT = Path("data/interim/tiles/correntoso-arauco")
OUTPUT = Path("data/interim/previews")
N_SAMPLES = 30

OUTPUT.mkdir(parents=True, exist_ok=True)
random.seed(42)

tiles = list(INPUT.glob("*.tif"))
sample = random.sample(tiles, min(N_SAMPLES, len(tiles)))

for tif in sample:
    with rasterio.open(tif) as src:
        data = src.read()   # shape (bands, h, w)
    # rasterio returns (C, H, W); PIL expects (H, W, C)
    img = np.transpose(data, (1, 2, 0))
    if img.shape[2] == 4:
        img = img[:, :, :3]   # drop alpha
    Image.fromarray(img).save(OUTPUT / f"{tif.stem}.png")

print(f"{len(sample)} previews saved to {OUTPUT}")
