"""Merge Roboflow's 70/15/15 splits of the oracular project into a single
``test/`` dir, dropping the auto-generated "root" category.
"""

import json
import shutil
from pathlib import Path

SRC = Path("data/processed/dataset_coco_oracular")
DST = Path("data/processed/dataset_coco_oracular_merged/test")
KEEP = {"building", "trees_and_bushes"}


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    merged_imgs: list[dict] = []
    merged_anns: list[dict] = []
    cats_out: list[dict] = []
    cat_remap: dict[int, int] = {}
    next_img_id = 1
    next_ann_id = 1

    for split in ("train", "valid", "test"):
        src_split = SRC / split
        coco = json.loads((src_split / "_annotations.coco.json").read_text())

        if not cats_out:
            for c in coco["categories"]:
                if c["name"] in KEEP:
                    new_id = len(cats_out) + 1
                    cats_out.append({**c, "id": new_id})
                    cat_remap[c["id"]] = new_id

        id_map: dict[int, int] = {}
        for img in coco["images"]:
            old_id = img["id"]
            img = {**img, "id": next_img_id}
            merged_imgs.append(img)
            shutil.copy2(src_split / img["file_name"], DST / img["file_name"])
            id_map[old_id] = next_img_id
            next_img_id += 1
        for a in coco["annotations"]:
            if a["category_id"] not in cat_remap:
                continue
            a = {
                **a,
                "id": next_ann_id,
                "image_id": id_map[a["image_id"]],
                "category_id": cat_remap[a["category_id"]],
            }
            merged_anns.append(a)
            next_ann_id += 1

    out = {
        "info": {},
        "licenses": [],
        "categories": cats_out,
        "images": merged_imgs,
        "annotations": merged_anns,
    }
    (DST / "_annotations.coco.json").write_text(json.dumps(out))
    print(f"merged: {len(merged_imgs)} imgs, {len(merged_anns)} anns, "
          f"cats={[c['name'] for c in cats_out]}")


if __name__ == "__main__":
    main()
