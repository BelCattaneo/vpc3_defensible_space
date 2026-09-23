"""Convert Roboflow COCO segmentation export to YOLO segmentation format.

Reads data/processed/dataset/{train,valid,test}/_annotations.coco.json
and writes data/processed/dataset_yolo/{images,labels}/{train,val,test}/
plus data.yaml for ultralytics.
"""

import json
import shutil
from pathlib import Path

SRC = Path("data/processed/dataset_coco_v5")
DST = Path("data/processed/dataset_yolo_v5")

# Roboflow uses 'valid' but ultralytics prefers 'val'
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

# Classes we keep (skip the 'defensible-space' root class Roboflow creates).
KEEP_CLASSES = {"building", "trees_and_bushes"}


def convert():
    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "labels").mkdir(parents=True, exist_ok=True)

    yolo_class_ids = {}  # populated from first split, reused in others

    for src_split, dst_split in SPLIT_MAP.items():
        src_dir = SRC / src_split
        if not src_dir.exists():
            print(f"skip {src_split} (missing)")
            continue

        img_out = DST / "images" / dst_split
        lbl_out = DST / "labels" / dst_split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        coco = json.loads((src_dir / "_annotations.coco.json").read_text())

        # assign contiguous yolo ids only for classes we keep
        if not yolo_class_ids:
            keep = [c for c in coco["categories"] if c["name"] in KEEP_CLASSES]
            keep.sort(key=lambda c: c["name"])
            yolo_class_ids.update({c["id"]: i for i, c in enumerate(keep)})

        # group annotations by image
        anns_by_img = {}
        for a in coco["annotations"]:
            if a["category_id"] not in yolo_class_ids:
                continue
            anns_by_img.setdefault(a["image_id"], []).append(a)

        for img in coco["images"]:
            src_path = src_dir / img["file_name"]
            if not src_path.exists():
                continue

            shutil.copy2(src_path, img_out / img["file_name"])

            w, h = img["width"], img["height"]
            lines = []
            for a in anns_by_img.get(img["id"], []):
                cls = yolo_class_ids[a["category_id"]]
                # segmentation is a list of polygons; each polygon is [x1,y1,x2,y2,...]
                for poly in a["segmentation"]:
                    norm = [
                        f"{(v / w if i % 2 == 0 else v / h):.6f}"
                        for i, v in enumerate(poly)
                    ]
                    lines.append(f"{cls} " + " ".join(norm))

            lbl_path = lbl_out / (Path(img["file_name"]).stem + ".txt")
            lbl_path.write_text("\n".join(lines))

        n_imgs = len(list(img_out.glob("*.jpg"))) + len(list(img_out.glob("*.png")))
        print(f"{dst_split}: {n_imgs} images written")

    # write data.yaml for ultralytics
    # need class names in order 0..N-1; recover from any split's coco categories
    coco = json.loads((SRC / "train" / "_annotations.coco.json").read_text())
    src_id2name = {c["id"]: c["name"] for c in coco["categories"]}
    names_in_order = [src_id2name[cid] for cid, _ in sorted(yolo_class_ids.items(), key=lambda kv: kv[1])]

    yaml_text = (
        f"path: {DST.absolute()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(names_in_order)}\n"
        f"names: {names_in_order}\n"
    )
    (DST / "data.yaml").write_text(yaml_text)
    print(f"data.yaml written with {len(names_in_order)} classes: {names_in_order}")


if __name__ == "__main__":
    convert()
