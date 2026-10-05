"""Build a filtered COCO dataset from the v6 merged export.

One script covers both historical derivatives. ``--labelers`` recibe las
cuentas de las anotadoras (los emails reales se mantienen fuera del repo,
leyendolos del mapping local ``reports/filenames_by_labeler.json``):

* dataset_coco_v7 (author-only, same splits as v6):
    python src/build_dataset.py \\
        --labelers <author-email> \\
        --output data/processed/dataset_coco_v7 \\
        --mode preserve-splits --include-unmapped

* dataset_coco_holdout_team (team-only, fused into a single test/):
    python src/build_dataset.py \\
        --labelers <team-email-1> <team-email-2> <team-email-3> \\
        --output data/processed/dataset_coco_holdout_team \\
        --mode merge-to-test
"""

import argparse
import json
import shutil
from pathlib import Path


def canon(name: str) -> str:
    """Strip Roboflow preprocessing suffix: 'foo_png.rf.<hash>.jpg' -> 'foo'."""
    stem = name
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    for ext in ("_png", ".png", ".jpg", ".jpeg"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem


def _load_labeler_sets(labeler_map: Path,
                       keep: set[str]) -> tuple[set[str], set[str]]:
    """Return (kept_canonical_names, all_canonical_names_in_jobs)."""
    by_labeler = json.loads(labeler_map.read_text())
    kept, all_names = set(), set()
    for lab, names in by_labeler.items():
        for n in names:
            c = canon(n)
            all_names.add(c)
            if lab in keep:
                kept.add(c)
    return kept, all_names


def _write_split(dst_split: Path, coco: dict,
                 kept_imgs: list[dict], kept_anns: list[dict]) -> None:
    (dst_split / "_annotations.coco.json").write_text(json.dumps({
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": kept_imgs,
        "annotations": kept_anns,
    }))


def _preserve_splits(src: Path, dst: Path,
                     kept: set[str], all_names: set[str],
                     include_unmapped: bool) -> None:
    """Mirror v6's train/valid/test under dst, keeping only matched images."""
    for split in ("train", "valid", "test"):
        src_split = src / split
        dst_split = dst / split
        dst_split.mkdir(parents=True, exist_ok=True)
        coco = json.loads((src_split / "_annotations.coco.json").read_text())
        kept_imgs = [
            img for img in coco["images"]
            if canon(img["file_name"]) in kept
               or (include_unmapped and canon(img["file_name"]) not in all_names)
        ]
        kept_ids = {img["id"] for img in kept_imgs}
        kept_anns = [a for a in coco["annotations"] if a["image_id"] in kept_ids]
        for img in kept_imgs:
            shutil.copy2(src_split / img["file_name"], dst_split / img["file_name"])
        _write_split(dst_split, coco, kept_imgs, kept_anns)
        print(f"  {split}: {len(kept_imgs)}/{len(coco['images'])} images, "
              f"{len(kept_anns)}/{len(coco['annotations'])} anns")


def _merge_to_test(src: Path, dst: Path, kept: set[str]) -> None:
    """Fuse all v6 splits into dst/test/, remapping IDs to avoid collisions."""
    test_dir = dst / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    merged_imgs, merged_anns = [], []
    categories: list[dict] = []
    next_img_id = next_ann_id = 1
    for split in ("train", "valid", "test"):
        coco = json.loads((src / split / "_annotations.coco.json").read_text())
        if not categories:
            categories = coco["categories"]
        id_map: dict[int, int] = {}
        for img in coco["images"]:
            if canon(img["file_name"]) not in kept:
                continue
            id_map[img["id"]] = next_img_id
            merged_imgs.append({**img, "id": next_img_id})
            shutil.copy2(src / split / img["file_name"],
                         test_dir / img["file_name"])
            next_img_id += 1
        for a in coco["annotations"]:
            if a["image_id"] in id_map:
                merged_anns.append({**a, "id": next_ann_id,
                                    "image_id": id_map[a["image_id"]]})
                next_ann_id += 1
    _write_split(test_dir, {"categories": categories}, merged_imgs, merged_anns)
    print(f"test: {len(merged_imgs)} images, {len(merged_anns)} anns")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path,
                   default=Path("data/processed/dataset_coco_v6"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--labeler-map", type=Path,
                   default=Path("reports/filenames_by_labeler.json"))
    p.add_argument("--labelers", nargs="+", required=True,
                   help="emails to keep")
    p.add_argument("--mode", choices=("preserve-splits", "merge-to-test"),
                   default="preserve-splits")
    p.add_argument("--include-unmapped", action="store_true",
                   help="preserve-splits only: also keep images not in any "
                        "annotation job (treated as author-labeled)")
    args = p.parse_args()

    kept, all_names = _load_labeler_sets(args.labeler_map, set(args.labelers))
    print(f"kept canonical names: {len(kept)} of {len(all_names)} mapped")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "preserve-splits":
        _preserve_splits(args.source, args.output, kept, all_names,
                         args.include_unmapped)
    else:
        _merge_to_test(args.source, args.output, kept)


if __name__ == "__main__":
    main()
