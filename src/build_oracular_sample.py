"""Pick a stratified random sample of held-out-team tiles for re-labeling.

Output goes to ``data/oracular_relabel/`` with a mix of barrio-norte and
correntoso-arauco tiles, plus a README with the labeling criteria.

Two-pass sampling: the first 10 per sector come from seed ``SEED1``, the
remaining ``EXTRA_PER_SECTOR`` from seed ``SEED2`` over the pool excluding
the first pass. This lets us grow the sample without discarding tiles the
user already started relabeling.
"""

import json
import random
import shutil
from pathlib import Path

SRC = Path("data/processed/dataset_coco_holdout_team/test")
DST = Path("data/oracular_relabel")
FIRST_PASS_PER_SECTOR = 10
EXTRA_PER_SECTOR = 15
SEED1 = 42
SEED2 = 43


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    coco = json.loads((SRC / "_annotations.coco.json").read_text())
    by_sector: dict[str, list[dict]] = {"barrio-norte": [], "correntoso-arauco": []}
    for img in coco["images"]:
        for sec, bucket in by_sector.items():
            if img["file_name"].startswith(sec):
                bucket.append(img)
                break

    first = random.Random(SEED1)
    second = random.Random(SEED2)
    picked: list[dict] = []
    for sec, imgs in by_sector.items():
        p1 = first.sample(imgs, min(FIRST_PASS_PER_SECTOR, len(imgs)))
        remaining = [i for i in imgs if i["id"] not in {x["id"] for x in p1}]
        p2 = second.sample(remaining, min(EXTRA_PER_SECTOR, len(remaining)))
        picked.extend(p1 + p2)
        print(f"{sec}: {len(p1)} + {len(p2)} = {len(p1) + len(p2)} "
              f"(de {len(imgs)} disponibles)")

    for img in picked:
        shutil.copy2(SRC / img["file_name"], DST / img["file_name"])

    (DST / "picked_filenames.txt").write_text(
        "\n".join(sorted(img["file_name"] for img in picked)) + "\n"
    )
    print(f"\nescribio {len(picked)} tiles a {DST}")


if __name__ == "__main__":
    main()
