"""Convert a sample of TIF tiles to PNG and stage them for Roboflow upload.

Supports multiple batches without repeating tiles across batches.
"""

import argparse
import random
from pathlib import Path

from PIL import Image

from mask_utils import read_tif_as_rgb

SECTORS = ("barrio-norte", "correntoso-arauco")
BASE = Path("data/processed/roboflow_upload")

DEFAULT_SAMPLES_PER_SECTOR = 25
DEFAULT_SEED = 42
DEFAULT_BATCH = "train"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-sector", type=int, default=DEFAULT_SAMPLES_PER_SECTOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--batch", type=str, default=DEFAULT_BATCH, help="output subfolder name"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = BASE / args.batch
    output.mkdir(parents=True, exist_ok=True)

    # Skip tiles already staged in any previous batch to avoid duplicates.
    already_used = {p.name for p in BASE.rglob("*.png")}
    random.seed(args.seed)

    total = 0
    for sector in SECTORS:
        tiles = sorted(Path(f"data/interim/tiles/{sector}").glob("*.tif"))
        candidates = [t for t in tiles if f"{sector}__{t.stem}.png" not in already_used]
        picked = random.sample(candidates, min(args.samples_per_sector, len(candidates)))
        for tif in picked:
            img = read_tif_as_rgb(tif)
            out = output / f"{sector}__{tif.stem}.png"
            Image.fromarray(img).save(out)
            total += 1

    print(f"{total} PNGs saved to {output}")


if __name__ == "__main__":
    main()
