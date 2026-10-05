"""Convert Roboflow COCO segmentation export to YOLO segmentation format.

Reads ``data/processed/dataset_coco_v6/{train,valid,test}/_annotations.coco.json``
and writes ``data/processed/dataset_yolo_v6/{images,labels}/{train,val,test}/``
plus ``data.yaml`` for ultralytics.
"""

import json
import shutil
import sys
from pathlib import Path


def _clean_dir(d: Path) -> None:
    """Vacia ``d`` manteniendo el directorio para evitar mezclar versiones."""
    if d.exists():
        for item in d.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

from constants import KEEP_CLASSES

# Override via CLI: ``python coco_to_yolo.py <coco_dir> <yolo_dir>``
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/dataset_coco_v6")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/dataset_yolo_v6")

# Roboflow uses "valid" but ultralytics prefers "val".
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def _build_yolo_class_ids(coco: dict) -> dict[int, int]:
    """Map COCO category ids to contiguous YOLO label ids (alphabetical over KEEP_CLASSES)."""
    keep = [c for c in coco["categories"] if c["name"] in KEEP_CLASSES]
    keep.sort(key=lambda c: c["name"])
    return {c["id"]: i for i, c in enumerate(keep)}


def _convert_split(src_dir: Path, dst_split: str, yolo_class_ids: dict[int, int]) -> None:
    """Copy images and write YOLO label files for a single split."""
    img_out = DST / "images" / dst_split
    lbl_out = DST / "labels" / dst_split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    # Vaciar el directorio evita heredar imagenes/labels de una conversion
    # anterior cuando SRC o DST apuntan a versiones distintas del dataset.
    _clean_dir(img_out)
    _clean_dir(lbl_out)

    coco = json.loads((src_dir / "_annotations.coco.json").read_text())

    anns_by_img: dict[int, list[dict]] = {}
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
            # segmentation is a list of polygons; each polygon is [x1, y1, x2, y2, ...]
            # Normalizar y clampear a [0, 1]: Roboflow emite ocasionalmente
            # coords apenas fuera del borde (p.ej. 1024.5 en una imagen 1024),
            # y la API de Ultralytics las rechaza.
            for poly in a["segmentation"]:
                norm = [
                    f"{min(max(v / w if i % 2 == 0 else v / h, 0.0), 1.0):.6f}"
                    for i, v in enumerate(poly)
                ]
                lines.append(f"{cls} " + " ".join(norm))

        lbl_path = lbl_out / (Path(img["file_name"]).stem + ".txt")
        lbl_path.write_text("\n".join(lines))

    n_imgs = len(list(img_out.glob("*.jpg"))) + len(list(img_out.glob("*.png")))
    print(f"{dst_split}: {n_imgs} images written")


def _write_data_yaml(yolo_class_ids: dict[int, int]) -> None:
    """Write the ultralytics ``data.yaml`` with class names in YOLO id order."""
    # Prefer train split, fall back a cualquiera presente para soportar
    # datasets test-only (p.ej. holdout_team).
    for split in ("train", "valid", "test"):
        candidate = SRC / split / "_annotations.coco.json"
        if candidate.exists():
            coco = json.loads(candidate.read_text())
            break
    else:
        raise FileNotFoundError(f"no split with annotations found under {SRC}")
    src_id2name = {c["id"]: c["name"] for c in coco["categories"]}
    names_in_order = [
        src_id2name[cid]
        for cid, _ in sorted(yolo_class_ids.items(), key=lambda kv: kv[1])
    ]

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


def main() -> None:
    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "labels").mkdir(parents=True, exist_ok=True)

    yolo_class_ids: dict[int, int] = {}
    for src_split, dst_split in SPLIT_MAP.items():
        src_dir = SRC / src_split
        if not src_dir.exists():
            print(f"skip {src_split} (missing)")
            continue

        if not yolo_class_ids:
            coco = json.loads((src_dir / "_annotations.coco.json").read_text())
            yolo_class_ids = _build_yolo_class_ids(coco)

        _convert_split(src_dir, dst_split, yolo_class_ids)

    _write_data_yaml(yolo_class_ids)


if __name__ == "__main__":
    main()
