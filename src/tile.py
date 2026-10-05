"""Tile orthophotos into 1024x1024 patches with 20% overlap."""

import argparse
from pathlib import Path

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as win_transform

TILE_SIZE = 1024
OVERLAP = 0.2
STEP = int(TILE_SIZE * (1.0 - OVERLAP))

# Skip tiles whose transparent alpha covers more than this fraction.
MAX_TRANSPARENT_FRACTION = 0.5

# Set to an int to process only the first N orthophotos; None for all.
LIMIT: int | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sector", required=True, help="sector name (folder under data/raw/)"
    )
    return parser.parse_args()


def _tile_one(tif: Path, out_dir: Path) -> None:
    """Write all valid 1024x1024 tiles of a single orthophoto."""
    with rasterio.open(tif) as src:
        meta = src.meta.copy()
        meta.update(width=TILE_SIZE, height=TILE_SIZE, compress="lzw")
        kept = 0
        skipped = 0
        if src.height < TILE_SIZE or src.width < TILE_SIZE:
            print(f"{tif.name}: skipped (ortofoto {src.width}x{src.height} "
                  f"smaller than tile size {TILE_SIZE}x{TILE_SIZE})")
            return
        # step-based origins plus one final origin per axis so the last
        # ~STEP px band at the south/east edge is not silently dropped.
        ys = list(range(0, src.height - TILE_SIZE + 1, STEP))
        if ys and ys[-1] != src.height - TILE_SIZE:
            ys.append(src.height - TILE_SIZE)
        xs = list(range(0, src.width - TILE_SIZE + 1, STEP))
        if xs and xs[-1] != src.width - TILE_SIZE:
            xs.append(src.width - TILE_SIZE)
        for y in ys:
            for x in xs:
                win = Window(x, y, TILE_SIZE, TILE_SIZE)
                data = src.read(window=win)
                if src.count == 4 and (data[3] == 0).mean() > MAX_TRANSPARENT_FRACTION:
                    skipped += 1
                    continue
                meta["transform"] = win_transform(win, src.transform)
                out = out_dir / f"{tif.stem}__x{x}_y{y}.tif"
                with rasterio.open(out, "w", **meta) as dst:
                    dst.write(data)
                kept += 1
        print(f"{tif.name}: {kept} tiles kept, {skipped} skipped (transparency)")


def main() -> None:
    args = _parse_args()
    input_dir = Path(f"data/raw/{args.sector}")
    output_dir = Path(f"data/interim/tiles/{args.sector}")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.tif"))
    if LIMIT:
        paths = paths[:LIMIT]

    for tif in paths:
        _tile_one(tif, output_dir)


if __name__ == "__main__":
    main()
