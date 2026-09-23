"""Convert a sample of TIF tiles to PNG and stage them for Roboflow upload.

Supports multiple batches without repeating tiles across batches.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--samples-per-sector", type=int, default=25)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch", type=str, default="train", help="output subfolder name")
args = parser.parse_args()

SECTORS = ["barrio-norte", "correntoso-arauco"]
BASE = Path("data/processed/roboflow_upload")
OUTPUT = BASE / args.batch
OUTPUT.mkdir(parents=True, exist_ok=True)

# collect names of tiles already staged in any previous batch to avoid duplicates
already_used = {p.name for p in BASE.rglob("*.png")}

random.seed(args.seed)

total = 0
for sector in SECTORS:
    tiles = sorted(Path(f"data/interim/tiles/{sector}").glob("*.tif"))
    # filter out tiles already staged
    candidates = [t for t in tiles if f"{sector}__{t.stem}.png" not in already_used]
    picked = random.sample(candidates, min(args.samples_per_sector, len(candidates)))
    for tif in picked:
        with rasterio.open(tif) as src:
            data = src.read()
        img = np.transpose(data, (1, 2, 0))
        if img.shape[2] == 4:
            img = img[:, :, :3]
        out = OUTPUT / f"{sector}__{tif.stem}.png"
        Image.fromarray(img).save(out)
        total += 1

print(f"{total} PNGs saved to {OUTPUT}")
