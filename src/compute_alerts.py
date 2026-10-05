"""Compute defensible-space violation alerts per tile.

For each input image, runs a segmentation model, extracts per-instance
masks for buildings and trees_and_bushes, and computes the minimum
distance from every building to the nearest tree/bush. Writes a JSON
with per-building status plus an overlay JPG for visual inspection.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

from constants import DEVICE
from models_io import load_mask2former, load_yolo, m2f_instance_masks

M2F_DIR = Path(os.environ.get("M2F_MODEL_DIR", "models/mask2former"))
YOLO_WEIGHTS = Path(os.environ.get("YOLO_WEIGHTS", "models/yolo_seg/weights/best.pt"))

# Compliance rule defaults.
DEFAULT_THRESHOLD_M = 2.0
# Fallback only: when the input is a GeoTIFF (the normal path), the real
# pixel size is read from the transform. The ortofotos del proyecto tienen
# 5 cm/px (20 px/m) sobre un CRS UTM, así que si cae al fallback este valor
# minimiza el error en las distancias calculadas.
DEFAULT_PIXELS_PER_METER = 20.0
DEFAULT_CONF_M2F = 0.7
DEFAULT_CONF_YOLO = 0.25

# Overlay drawing constants (RGB, plus per-usage alpha).
COLOR_OK = (34, 189, 90)
COLOR_VIOLATION = (232, 65, 24)
COLOR_TREES = (34, 189, 90)
ALPHA_TREES = 70
ALPHA_BUILDING = 150
OVERLAY_FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"
OVERLAY_FONT_SIZE = 20

# Polygon extraction constants.
POLY_SIMPLIFY_EPS_PX = 2.0   # ~10 cm at 5 cm/px
DANGER_ZONE_MIN_AREA_PX = 4
AT_EDGE_MARGIN_PX = 25   # 25 px at 5 cm/px = 1,25 m, cubre el umbral de 2 m


# ---------------------------------------------------------------------------
# per-instance mask predictions
# ---------------------------------------------------------------------------

def predict_m2f(image, model, processor, id2name,
                conf_threshold: float = DEFAULT_CONF_M2F,
                ) -> dict[str, list[dict]]:
    """Return ``{class_name: [{mask, score}]}`` for one image via Mask2Former."""
    target_hw = (image.size[1], image.size[0])
    per_class: dict[str, list[dict]] = {}
    for label_id, bin_mask, score in m2f_instance_masks(
        model, processor, image, target_hw, conf_threshold
    ):
        if bin_mask.sum() == 0:
            continue
        per_class.setdefault(id2name[label_id], []).append({
            "mask": bin_mask,
            "score": float(score),
        })
    return per_class


def predict_yolo(image, model, class_names,
                 conf_threshold: float = DEFAULT_CONF_YOLO,
                 imgsz: int | None = None,
                 ) -> dict[str, list[dict]]:
    """Return ``{class_name: [{mask, score}]}`` for one image via YOLO.

    ``imgsz`` must match the resolution the model was trained at. YOLO
    tolerates different sizes but mask quality degrades sharply when the
    inference ``imgsz`` is larger than training (e.g. YOLOv8l v12 was
    trained at 640 and inferring at 1024 produces inflated blob masks that
    cover entire vegetation patches). Default here reads the training
    ``imgsz`` from the model checkpoint.
    """
    arr = np.array(image)
    h, w = arr.shape[:2]
    if imgsz is None:
        imgsz = int(model.overrides.get("imgsz")
                    or getattr(model, "args", {}).get("imgsz", 640))
    res = model.predict(arr, conf=conf_threshold, imgsz=imgsz,
                        device=DEVICE, verbose=False)[0]
    per_class: dict[str, list[dict]] = {}
    if res.masks is None:
        return per_class
    masks = res.masks.data.cpu().numpy()   # (N, Hp, Wp) at YOLO internal size
    cls = res.boxes.cls.cpu().numpy().astype(int)
    scores = res.boxes.conf.cpu().numpy().astype(float)
    for m, c, s in zip(masks, cls, scores):
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)
        per_class.setdefault(class_names[c], []).append({
            "mask": (m > 0.5).astype(np.uint8),
            "score": float(s),
        })
    return per_class


# ---------------------------------------------------------------------------
# mask geometry helpers
# ---------------------------------------------------------------------------

def compute_distances(building_masks: list[np.ndarray],
                      tree_union: np.ndarray | None) -> list[float | None]:
    """Minimum pixel distance from each building to the nearest tree.

    ``tree_union`` is the OR of all tree masks in the tile, or None when no
    trees were detected. Passing the union in avoids rebuilding it twice
    in the caller. ``None`` significa que no hay vegetación detectada contra
    la cual medir (en vez de ``float('inf')``, que no es JSON válido).
    """
    if not building_masks:
        return []
    if tree_union is None or not tree_union.any():
        return [None] * len(building_masks)
    edt = distance_transform_edt(~tree_union)
    distances: list[float | None] = []
    for bmask in building_masks:
        if bmask.sum() == 0:
            distances.append(None)
        else:
            distances.append(float(edt[bmask.astype(bool)].min()))
    return distances


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    """Return ``[x, y, w, h]`` of a binary mask (or all-zeros when empty).

    Ancho y alto incluyen el píxel final: una máscara de 1 px tiene w=h=1.
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def polygon_from_mask(mask: np.ndarray,
                      simplify_eps_px: float = POLY_SIMPLIFY_EPS_PX,
                      ) -> list[list[int]]:
    """Largest external contour of ``mask`` as a simplified ``[[x, y], ...]`` list.

    ``simplify_eps_px`` is the Ramer-Douglas-Peucker tolerance in pixels; 2 px
    (~10 cm at 5 cm/px) keeps building corners while dropping sub-pixel noise.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(largest, simplify_eps_px, closed=True)
    return [[int(pt[0][0]), int(pt[0][1])] for pt in simplified]


def is_at_edge(bbox: list[int], w: int, h: int,
               margin: int = AT_EDGE_MARGIN_PX) -> bool:
    """True when the bbox is within ``margin`` pixels of any tile border.

    At 5 cm/px, ``margin=25`` == 1,25 m, lo que cubre el umbral de 2 m con
    margen y marca cualquier caso en el que un árbol fuera del tile podría
    cambiar el resultado de la clasificación.
    """
    x, y, bw, bh = bbox
    return (x <= margin or y <= margin
            or (x + bw) >= (w - margin) or (y + bh) >= (h - margin))


def compute_danger_zones(building_masks: list[np.ndarray],
                         tree_masks: list[np.ndarray],
                         tree_scores: list[float],
                         threshold_px: float,
                         skip: list[bool] | None = None,
                         ) -> list[list[dict]]:
    """Trees within ``threshold_px`` of each building, as scored polygons.

    Each building yields zero or more ``{"polygon": [[x,y],...], "tree_score": float}``
    dicts (only pieces of vegetation that actually fall inside the buffer).
    The score is the max score among tree instances intersecting that zone.
    Small specks under ``DANGER_ZONE_MIN_AREA_PX`` are skipped.

    ``skip`` opcional: lista de booleanos del mismo largo que ``building_masks``;
    cuando ``skip[i]`` es True se devuelve lista vacía para ese edificio sin
    calcular la dilatación (la parte más cara del pipeline). Útil para no
    computar zonas de peligro de edificios ya marcados como compliant.
    """
    tree_union = _union_masks(tree_masks)
    if tree_union is None or not tree_union.any():
        return [[] for _ in building_masks]

    radius = round(threshold_px)
    if radius <= 0:
        return [[] for _ in building_masks]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius + 1, 2 * radius + 1))
    zones: list[list[dict]] = []
    for idx, bmask in enumerate(building_masks):
        if skip is not None and skip[idx]:
            zones.append([])
            continue
        dilated = cv2.dilate(bmask.astype(np.uint8), kernel)
        danger = np.logical_and(dilated.astype(bool),
                                tree_union.astype(bool)).astype(np.uint8)
        if danger.sum() == 0:
            zones.append([])
            continue
        contours, _ = cv2.findContours(
            danger, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )
        polys = []
        for c in contours:
            if cv2.contourArea(c) < DANGER_ZONE_MIN_AREA_PX:
                continue
            cmask = np.zeros_like(danger, dtype=np.uint8)
            cv2.drawContours(cmask, [c], -1, 1, -1)
            cbool = cmask.astype(bool)
            best = 0.0
            for tmask, tscore in zip(tree_masks, tree_scores):
                if np.logical_and(cbool, tmask.astype(bool)).any():
                    best = max(best, tscore)
            simp = cv2.approxPolyDP(c, POLY_SIMPLIFY_EPS_PX, closed=True)
            polys.append({
                "polygon": [[int(p[0][0]), int(p[0][1])] for p in simp],
                "tree_score": round(float(best), 4),
            })
        zones.append(polys)
    return zones


def _union_masks(masks: list[np.ndarray]) -> np.ndarray | None:
    """Elementwise OR of a list of binary masks, or None when the list is empty."""
    if not masks:
        return None
    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        out |= m.astype(bool)
    return out


# ---------------------------------------------------------------------------
# overlay rendering
# ---------------------------------------------------------------------------

def draw_overlay(image: Image.Image,
                 tree_masks: list[np.ndarray],
                 building_records: list[dict]) -> Image.Image:
    """Render an RGB overlay: green trees + red/green buildings + distance labels."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    tree_union = _union_masks(tree_masks)
    if tree_union is not None:
        rgba = np.zeros((*tree_union.shape, 4), dtype=np.uint8)
        rgba[tree_union] = (*COLOR_TREES, ALPHA_TREES)
        overlay = Image.alpha_composite(overlay, Image.fromarray(rgba))

    for rec in building_records:
        color = COLOR_VIOLATION if not rec["compliant"] else COLOR_OK
        mask = rec["_mask"]
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask.astype(bool)] = (*color, ALPHA_BUILDING)
        overlay = Image.alpha_composite(overlay, Image.fromarray(rgba))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(OVERLAY_FONT_PATH, OVERLAY_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    for rec in building_records:
        x, y, _, _ = rec["bbox"]
        dist_m = rec.get("distance_m")
        label = "sin veg" if dist_m is None else f"{dist_m:.1f} m"
        draw.text((x + 4, y + 4), label,
                  fill="white", font=font,
                  stroke_width=2, stroke_fill="black")
    return out


# ---------------------------------------------------------------------------
# per-tile pipeline
# ---------------------------------------------------------------------------

def _resolve_px_per_meter(image_path: Path, fallback: float) -> float:
    """Return the real pixels-per-meter from the TIFF transform, or ``fallback``.

    Uses the absolute value of the transform's x pixel size, which for a
    projected CRS in metres (UTM, Web Mercator) is the pixel size in metres.
    If the CRS is geographic (lat/lon in degrees) or the file is not a GeoTIFF,
    ``fallback`` is returned.
    """
    try:
        with rasterio.open(image_path) as src:
            crs = src.crs
            if crs is None or crs.is_geographic:
                return fallback
            px_size = abs(src.transform.a)
            if px_size <= 0:
                return fallback
            return 1.0 / px_size
    except (rasterio.RasterioIOError, OSError, ValueError):
        return fallback


def _predict(image: Image.Image, model_kind: str, backends: dict,
             conf_threshold: float) -> dict[str, list[dict]]:
    """Route to the right per-model predictor."""
    if model_kind == "m2f":
        processor, model, id2name = backends["m2f"]
        return predict_m2f(image, model, processor, id2name, conf_threshold)
    model, class_names = backends["yolo"]
    return predict_yolo(image, model, class_names, conf_threshold)


def _build_records(building_masks: list[np.ndarray],
                   building_scores: list[float],
                   distances_px: list[float | None],
                   danger_zones: list[list[dict]],
                   threshold_m: float, px_per_meter: float,
                   w: int, h: int) -> list[dict]:
    """Bundle per-building geometry + compliance into the JSON schema.

    ``distance_m``/``distance_px`` quedan en ``None`` cuando no hay vegetación
    contra la cual medir (JSON válido en vez de ``Infinity``), y en ese caso
    el edificio se marca como compliant.
    """
    records = []
    for i, (mask, score, dist_px, dzones) in enumerate(
        zip(building_masks, building_scores, distances_px, danger_zones)
    ):
        if dist_px is None:
            dist_m = None
            compliant = True
        else:
            dist_m = dist_px / px_per_meter
            compliant = dist_m >= threshold_m
        bbox = bbox_from_mask(mask)
        records.append({
            "id": i,
            "score": round(float(score), 4),
            "distance_m": None if dist_m is None else round(dist_m, 2),
            "distance_px": None if dist_px is None else round(dist_px, 1),
            "compliant": compliant,
            "at_edge": is_at_edge(bbox, w, h),
            "bbox": bbox,
            "polygon_px": polygon_from_mask(mask),
            # only store danger zones for actual violations (saves file size)
            "danger_zones_px": dzones if not compliant else [],
            "_mask": mask,   # dropped before writing JSON
        })
    return records


def process_tile(image_path: Path, model_kind: str, backends: dict,
                 threshold_m: float, px_per_meter: float,
                 conf_threshold: float,
                 ) -> tuple[dict, Image.Image]:
    """Full per-tile pipeline: predict → distances → danger zones → JSON + overlay.

    When the input is a georeferenced TIFF with a projected CRS (metres),
    ``px_per_meter`` is overridden by the actual pixel size read from the
    transform. The CLI value is only kept as fallback for non-georef inputs
    (e.g. JPG tiles).
    """
    px_per_meter = _resolve_px_per_meter(image_path, px_per_meter)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    preds = _predict(image, model_kind, backends, conf_threshold)

    # Drop empty masks up front so bbox/at_edge/polygon do not see garbage.
    building_entries = [e for e in preds.get("building", [])
                        if e["mask"].sum() > 0]
    tree_entries = [e for e in preds.get("trees_and_bushes", [])
                    if e["mask"].sum() > 0]
    building_masks = [e["mask"] for e in building_entries]
    building_scores = [e["score"] for e in building_entries]
    tree_masks = [e["mask"] for e in tree_entries]
    tree_scores = [e["score"] for e in tree_entries]
    tree_union = _union_masks(tree_masks)

    distances_px = compute_distances(building_masks, tree_union)
    # Pre-compute compliance to skip danger-zone work for already-compliant
    # buildings (the dilation is the most expensive step per building).
    threshold_px = threshold_m * px_per_meter
    skip = [d is None or d >= threshold_px for d in distances_px]
    danger_zones = compute_danger_zones(building_masks, tree_masks, tree_scores,
                                        threshold_px, skip=skip)
    records = _build_records(building_masks, building_scores,
                             distances_px, danger_zones,
                             threshold_m, px_per_meter, w, h)

    trees_out = []
    for i, (tmask, tscore) in enumerate(zip(tree_masks, tree_scores)):
        poly = polygon_from_mask(tmask)
        if len(poly) < 3:
            continue
        trees_out.append({
            "id": i,
            "score": round(float(tscore), 4),
            "polygon_px": poly,
        })

    result = {
        "tile": image_path.name,
        "model": model_kind,
        "threshold_m": threshold_m,
        "conf_threshold": conf_threshold,
        "n_buildings": len(records),
        "n_trees_masks": len(tree_masks),
        "n_violations": sum(1 for r in records if not r["compliant"]),
        "buildings": [{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in records],
        "trees": trees_out,
    }
    overlay = draw_overlay(image, tree_masks, records)
    return result, overlay


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _load_backends(model_kind: str) -> dict:
    """Load the model and any needed metadata for a given backend."""
    if model_kind == "m2f":
        processor, model = load_mask2former(M2F_DIR)
        id2name = {int(k): v for k, v in model.config.id2label.items()}
        return {"m2f": (processor, model, id2name)}
    yolo, class_names = load_yolo(YOLO_WEIGHTS)
    return {"yolo": (yolo, class_names)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["m2f", "yolo"], default="m2f")
    parser.add_argument("--input", type=Path, required=True,
                        help="image file or directory")
    parser.add_argument("--output", type=Path, default=Path("reports/alerts"))
    parser.add_argument("--threshold-m", type=float, default=DEFAULT_THRESHOLD_M)
    parser.add_argument("--pixels-per-meter", type=float,
                        default=DEFAULT_PIXELS_PER_METER)
    parser.add_argument("--conf-threshold", type=float, default=None,
                        help="detection confidence threshold; "
                             "defaults to 0.7 for m2f and 0.25 for yolo")
    parser.add_argument("--pattern", default="*.jpg",
                        help="glob when input is a directory")
    return parser.parse_args()


def _resolve_conf(explicit: float | None, model_kind: str) -> float:
    """Choose an appropriate default confidence when the CLI did not set one."""
    if explicit is not None:
        return explicit
    return DEFAULT_CONF_M2F if model_kind == "m2f" else DEFAULT_CONF_YOLO


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    conf_threshold = _resolve_conf(args.conf_threshold, args.model)
    backends = _load_backends(args.model)

    tiles = (sorted(args.input.glob(args.pattern))
             if args.input.is_dir() else [args.input])

    all_results = []
    for tile in tiles:
        result, overlay = process_tile(
            tile, args.model, backends,
            args.threshold_m, args.pixels_per_meter,
            conf_threshold=conf_threshold,
        )
        (args.output / f"{tile.stem}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
        overlay.save(args.output / f"{tile.stem}_overlay.jpg", quality=88)
        all_results.append(result)
        print(f"{tile.name}: {result['n_buildings']} buildings, "
              f"{result['n_violations']} violations")

    total_buildings = sum(r["n_buildings"] for r in all_results)
    total_violations = sum(r["n_violations"] for r in all_results)
    summary = {
        "n_tiles": len(all_results),
        "total_buildings": total_buildings,
        "total_violations": total_violations,
        "violation_rate": (round(total_violations / total_buildings, 3)
                           if total_buildings else None),
        "threshold_m": args.threshold_m,
        "model": args.model,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    print(f"\nwrote {len(tiles)} tiles to {args.output}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
