"""Tile orthophotos into 1024x1024 patches with 20% overlap."""

import argparse
from pathlib import Path

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as win_transform

TILE_SIZE = 1024
STEP = int(TILE_SIZE * 0.8)   # 20% overlap
LIMIT = None                  # None to process all orthophotos

parser = argparse.ArgumentParser()
parser.add_argument("--sector", required=True, help="sector name (folder under data/raw/)")
args = parser.parse_args()

INPUT = Path(f"data/raw/{args.sector}")
OUTPUT = Path(f"data/interim/tiles/{args.sector}")

OUTPUT.mkdir(parents=True, exist_ok=True)

paths = sorted(INPUT.glob("*.tif"))
if LIMIT:
    paths = paths[:LIMIT]

for tif in paths:
    with rasterio.open(tif) as src:
        meta = src.meta.copy()
        meta.update(width=TILE_SIZE, height=TILE_SIZE, compress="lzw")
        kept = 0
        skipped = 0
        for y in range(0, src.height - TILE_SIZE + 1, STEP):
            for x in range(0, src.width - TILE_SIZE + 1, STEP):
                win = Window(x, y, TILE_SIZE, TILE_SIZE)
                data = src.read(window=win)
                # skip tiles that are more than 50% transparent
                if src.count == 4 and (data[3] == 0).mean() > 0.5:
                    skipped += 1
                    continue
                meta["transform"] = win_transform(win, src.transform)
                out = OUTPUT / f"{tif.stem}__x{x}_y{y}.tif"
                with rasterio.open(out, "w", **meta) as dst:
                    dst.write(data)
                kept += 1
        print(f"{tif.name}: {kept} tiles kept, {skipped} skipped (transparency)")
