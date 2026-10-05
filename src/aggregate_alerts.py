"""Aggregate per-tile alert JSONs into a single georeferenced GeoJSON.

Reads the JSONs produced by ``compute_alerts.py``, opens the matching GeoTIFF
tile with rasterio to recover the geotransform and CRS, and converts each
building polygon and danger zone from pixel coordinates to WGS84.

Output: a single FeatureCollection with one Polygon feature per building
plus one per danger zone, carrying properties (distance_m, compliant,
tile_id, at_edge, model, danger_area_m2).
"""

import argparse
import json
import math
from pathlib import Path

import rasterio
from rasterio.transform import xy as pixel_to_xy
from rasterio.warp import transform as warp_transform
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

# Municipal defensible-space rule (Villa La Angostura Zone 1): trees < 2 m
# from a structure counts as a violation. Used as the fallback when no
# per-tile JSON declares its own threshold.
DEFAULT_THRESHOLD_M = 2.0

# Dedup settings. The same building appears recortado en dos o tres tiles
# porque los tiles se superponen un 20 %. Fusionamos solo cuando dos
# polígonos efectivamente se solapan (IoU por encima del umbral); detecciones
# que apenas se tocan pero no se solapan son casas distintas o la misma casa
# capturada junto a un pedazo de suelo adyacente, y no deben colapsarse.
DEDUP_IOU_THRESHOLD = 0.3

# Fuel-mass reclassification: a violation with less than this much nearby
# vegetation gets demoted to compliant (isolated small tree, not a real fuel).
DEFAULT_MIN_DANGER_AREA_M2 = 3.0

# Rough Villa La Angostura latitude, used as a fallback for the sqrdeg to sqm
# conversion when the feature collection turns out to be empty.
FALLBACK_LATITUDE = -40.75
LAT_M_PER_DEG = 111_320.0

MIN_POLYGON_POINTS = 3
MAX_LAT_SAMPLES = 200


# ---------------------------------------------------------------------------
# tile lookup + coordinate conversion
# ---------------------------------------------------------------------------

def resolve_tif_path(tile_stem: str, tiles_root: Path,
                     sector_names: list[str] | None = None) -> Path | None:
    """Find the source GeoTIFF matching this tile stem.

    Two supported stem shapes:

    (a) Roboflow-exported jpg (post-labeling pipeline):
        ``<sector>__orthophoto_<uuid>__x<X>_y<Y>_png.rf.<hash>``
        -> ``tiles_root/<sector>/orthophoto_<uuid>__x<X>_y<Y>.tif``

    (b) Raw TIF passed directly to ``compute_alerts`` (deployment path):
        ``orthophoto_<uuid>__x<X>_y<Y>``
        -> ``tiles_root/<any sector>/orthophoto_<uuid>__x<X>_y<Y>.tif``
    """
    if sector_names is None:
        sector_names = sorted(p.name for p in tiles_root.iterdir() if p.is_dir())

    # Case (a): sector prefix present.
    for sector in sector_names:
        prefix = f"{sector}__"
        if tile_stem.startswith(prefix):
            rest = tile_stem[len(prefix):]
            if ".rf." in rest:
                rest = rest.split(".rf.")[0]
            rest = rest.removesuffix("_png")
            candidate = tiles_root / sector / f"{rest}.tif"
            if candidate.exists():
                return candidate
            break   # prefix matched but the derived .tif is gone; fall through to (b)

    # Case (b): search across sectors.
    for sector in sector_names:
        candidate = tiles_root / sector / f"{tile_stem}.tif"
        if candidate.exists():
            return candidate
    return None


def polygon_to_geo(polygon_px: list[list[int]], transform,
                   src_crs) -> list[list[float]]:
    """Convert a pixel-coord polygon to ``[lon, lat]`` pairs in WGS84."""
    xs, ys = [], []
    for x_px, y_px in polygon_px:
        x, y = pixel_to_xy(transform, y_px, x_px)
        xs.append(x)
        ys.append(y)
    if src_crs is not None and src_crs.to_string() != "EPSG:4326":
        lon, lat = warp_transform(src_crs, "EPSG:4326", xs, ys)
    else:
        lon, lat = xs, ys
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------

def _polygon_feature(coords: list[list[float]], properties: dict) -> dict:
    """Return a closed-ring GeoJSON Polygon Feature con geometría válida.

    Cierra el anillo si hace falta y pasa por ``make_valid`` para arreglar
    auto-intersecciones y otros defectos que GIS consumers rechazarían.
    """
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    geom = {"type": "Polygon", "coordinates": [coords]}
    g = shape(geom)
    if not g.is_valid:
        g = make_valid(g)
    return {
        "type": "Feature",
        "geometry": mapping(g),
        "properties": properties,
    }


def build_feature(building: dict, tile_id: str, model: str,
                  transform, src_crs) -> dict | None:
    """Turn one building record into a georeferenced GeoJSON Feature."""
    poly_px = building.get("polygon_px", [])
    if len(poly_px) < MIN_POLYGON_POINTS:
        return None
    coords = polygon_to_geo(poly_px, transform, src_crs)
    # Scores of the trees that triggered each danger zone of this building.
    # Used by the map's vegetation slider to re-evaluate compliance in live:
    # a building is "really" a violation only while at least one of these
    # scores stays above the current vegetation threshold.
    danger_tree_scores = []
    for zone in building.get("danger_zones_px", []):
        if isinstance(zone, dict) and zone.get("tree_score") is not None:
            danger_tree_scores.append(round(float(zone["tree_score"]), 4))
    return _polygon_feature(coords, {
        "kind": "building",
        "tile_id": tile_id,
        "building_id": building["id"],
        "distance_m": building["distance_m"],
        "compliant": building["compliant"],
        "at_edge": building["at_edge"],
        "model": model,
        "score": building.get("score"),
        "danger_tree_scores": danger_tree_scores,
    })


def build_danger_zone_features(building: dict, tile_id: str, model: str,
                               transform, src_crs) -> list[dict]:
    """One Feature per danger zone polygon linked to a violation building.

    Accepts both the old shape (list of polygons) and the new one (list of
    ``{"polygon", "tree_score"}``). ``tree_score`` becomes the feature's
    ``score`` so the UI slider for vegetation can filter the zone by the
    confidence of the tree instance that generated it.
    """
    features = []
    for zone in building.get("danger_zones_px", []):
        if isinstance(zone, dict):
            poly = zone.get("polygon", [])
            tree_score = zone.get("tree_score")
        else:
            poly = zone
            tree_score = building.get("score")
        if len(poly) < MIN_POLYGON_POINTS:
            continue
        coords = polygon_to_geo(poly, transform, src_crs)
        features.append(_polygon_feature(coords, {
            "kind": "danger_zone",
            "tile_id": tile_id,
            "building_id": building["id"],
            "distance_m": building["distance_m"],
            "compliant": False,
            "at_edge": building["at_edge"],
            "model": model,
            "score": tree_score,
        }))
    return features


def build_tree_feature(tree: dict, tile_id: str, model: str,
                       transform, src_crs) -> dict | None:
    """Turn one tree record into a georeferenced GeoJSON Feature."""
    poly_px = tree.get("polygon_px", [])
    if len(poly_px) < MIN_POLYGON_POINTS:
        return None
    coords = polygon_to_geo(poly_px, transform, src_crs)
    return _polygon_feature(coords, {
        "kind": "tree",
        "tile_id": tile_id,
        "tree_id": tree["id"],
        "model": model,
        "score": tree.get("score"),
    })


# ---------------------------------------------------------------------------
# dedup: merge overlapping OR touching building polygons
# ---------------------------------------------------------------------------

def _shape_valid(feature: dict):
    """Return a valid shapely geometry (repaired if invalid)."""
    g = shape(feature["geometry"])
    return g if g.is_valid else make_valid(g)


def _find_root(parent: list[int], a: int) -> int:
    """Union-find lookup with path compression."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _union(parent: list[int], a: int, b: int) -> None:
    """Union-find union operation."""
    ra, rb = _find_root(parent, a), _find_root(parent, b)
    if ra != rb:
        parent[ra] = rb


def _group_overlapping(polys, iou_threshold: float) -> list[list[int]]:
    """Return groups of indices that should merge into one building.

    Two polygons se fusionan solo cuando su IoU es mayor o igual a
    ``iou_threshold``. El STRtree ya restringe la query a candidatos cuyo
    bounding box intersecta, así que la cuadrática queda sobre pares que
    tienen chance real de solaparse.
    """
    n = len(polys)
    parent = list(range(n))
    tree = STRtree(polys)
    for i, poly_i in enumerate(polys):
        for j in tree.query(poly_i):
            j = int(j)
            if j <= i:
                continue
            poly_j = polys[j]
            inter = poly_i.intersection(poly_j).area
            if inter == 0:
                continue
            uni = poly_i.union(poly_j).area
            if uni > 0 and (inter / uni) >= iou_threshold:
                _union(parent, i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(_find_root(parent, i), []).append(i)
    return list(groups.values())


def _merge_group(buildings: list[dict], polys, group: list[int],
                 threshold_m: float) -> dict | None:
    """Merge one group of building features into a single Feature."""
    if len(group) == 1:
        return buildings[group[0]]
    merged_poly = polys[group[0]]
    for k in group[1:]:
        merged_poly = merged_poly.union(polys[k])
    if merged_poly.is_empty or merged_poly.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    props = [buildings[k]["properties"] for k in group]
    # distance_m puede ser None cuando no habia vegetacion en el tile; al
    # fusionar ignoramos los None y nos quedamos con la minima real.
    dists = [p["distance_m"] for p in props if p.get("distance_m") is not None]
    min_dist = min(dists) if dists else None
    any_not_edge = any(not p.get("at_edge", False) for p in props)
    scores = [p.get("score") for p in props if p.get("score") is not None]
    max_score = max(scores) if scores else None
    merged_danger_scores: list[float] = []
    for p in props:
        merged_danger_scores.extend(p.get("danger_tree_scores") or [])
    return {
        "type": "Feature",
        "geometry": mapping(merged_poly),
        "properties": {
            "kind": "building",
            "tile_id": ",".join(sorted({p["tile_id"] for p in props})),
            "building_id": min(p["building_id"] for p in props),
            "distance_m": None if min_dist is None else round(min_dist, 2),
            "compliant": min_dist is None or min_dist >= threshold_m,
            "at_edge": not any_not_edge,
            "model": props[0]["model"],
            "score": max_score,
            "danger_tree_scores": merged_danger_scores,
            "merged_from": len(group),
        },
    }


def _merge_tree_group(trees: list[dict], polys, group: list[int]) -> dict | None:
    """Merge a group of tree features (overlapping or touching) into one."""
    if len(group) == 1:
        return trees[group[0]]
    merged_poly = polys[group[0]]
    for k in group[1:]:
        merged_poly = merged_poly.union(polys[k])
    if merged_poly.is_empty or merged_poly.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    props = [trees[k]["properties"] for k in group]
    scores = [p.get("score") for p in props if p.get("score") is not None]
    return {
        "type": "Feature",
        "geometry": mapping(merged_poly),
        "properties": {
            "kind": "tree",
            "tile_id": ",".join(sorted({p["tile_id"] for p in props})),
            "model": props[0]["model"],
            "score": max(scores) if scores else None,
            "merged_from": len(group),
        },
    }


def deduplicate(features: list[dict],
                iou_threshold: float = DEDUP_IOU_THRESHOLD,
                threshold_m: float = DEFAULT_THRESHOLD_M) -> list[dict]:
    """Merge overlapping building and tree polygons across tile borders.

    Each building appears recortado en dos o tres tiles vecinos por el
    solapamiento del 20 %. Fusionamos solo cuando la IoU entre dos polígonos
    supera ``iou_threshold``, lo que capta el solape real entre recortes del
    mismo edificio y descarta casas vecinas que apenas se tocan. Las
    detecciones de árboles se agrupan con el mismo criterio. Las zonas de
    peligro no se tocan porque están ligadas a un edificio específico.
    """
    if not features:
        return []
    buildings = [f for f in features
                 if f["properties"].get("kind", "building") == "building"]
    trees = [f for f in features if f["properties"].get("kind") == "tree"]
    pass_through = [f for f in features
                    if f["properties"].get("kind") not in ("building", "tree")]

    merged: list[dict] = []
    if buildings:
        polys = [_shape_valid(f) for f in buildings]
        for group in _group_overlapping(polys, iou_threshold):
            f = _merge_group(buildings, polys, group, threshold_m)
            if f is not None:
                merged.append(f)

    merged_trees: list[dict] = []
    if trees:
        tpolys = [_shape_valid(f) for f in trees]
        for group in _group_overlapping(tpolys, iou_threshold):
            f = _merge_tree_group(trees, tpolys, group)
            if f is not None:
                merged_trees.append(f)

    return merged + merged_trees + pass_through


# ---------------------------------------------------------------------------
# fuel-mass reclassification
# ---------------------------------------------------------------------------

def _sqrdeg_to_sqm(sample_lat: float) -> float:
    """Convert square-degrees to square-metres at a given latitude.

    1 sqrdeg ~ (111 km) * (111 km * cos(lat)) at that latitude. Small area
    approximation, good enough for a few km-scale sectors.
    """
    m_per_deg_lon = LAT_M_PER_DEG * math.cos(math.radians(sample_lat))
    return LAT_M_PER_DEG * m_per_deg_lon


def _mean_lat(features: list[dict]) -> float:
    """Mean latitude of the first N features (for the sqrdeg to sqm factor)."""
    lats = []
    for f in features[:MAX_LAT_SAMPLES]:
        geom = f["geometry"]
        if geom["type"] == "Polygon":
            ring = geom["coordinates"][0]
        elif geom["type"] == "MultiPolygon":
            ring = geom["coordinates"][0][0]
        else:
            continue
        lats.append(sum(p[1] for p in ring) / len(ring))
    return sum(lats) / len(lats) if lats else FALLBACK_LATITUDE


def _split_by_kind(features: list[dict]) -> tuple[list, list[int]]:
    """Return (danger_polys, building_indices)."""
    danger_polys = []
    building_indices = []
    for i, f in enumerate(features):
        kind = f["properties"].get("kind", "building")
        if kind == "danger_zone":
            danger_polys.append(_shape_valid(f))
        elif kind == "building":
            building_indices.append(i)
    return danger_polys, building_indices


def reclassify_by_danger_volume(features: list[dict],
                                min_danger_area_m2: float = DEFAULT_MIN_DANGER_AREA_M2,
                                ) -> None:
    """Cross-tile consistency + fuel-mass pass, in place.

    For every building we look up every danger zone polygon that intersects
    it, sum their area in square metres, and store it as ``danger_area_m2``.

    Two effects on the ``compliant`` flag:
      - a compliant building whose danger zones total >= ``min_danger_area_m2``
        becomes a violation (cross-tile: trees in another tile really were
        close enough, and there is enough of them to matter);
      - a violation whose danger zones total < ``min_danger_area_m2`` becomes
        compliant (the nearby vegetation is a single small crown, not
        enough continuous fuel to justify the alert).
    """
    danger_polys, building_indices = _split_by_kind(features)
    if not building_indices:
        return
    sqrdeg_to_sqm = _sqrdeg_to_sqm(_mean_lat(features))
    tree = STRtree(danger_polys) if danger_polys else None

    n_promoted = n_demoted = 0
    for i in building_indices:
        f = features[i]
        b = _shape_valid(f)
        area_sqdeg = 0.0
        if tree is not None:
            # Unimos las danger zones que intersectan antes de medir el
            # área: sin esto el solape entre zonas de tiles adyacentes se
            # contaría varias veces y el total podría exceder el área real.
            intersecting = [danger_polys[int(j)]
                            for j in tree.query(b)
                            if danger_polys[int(j)].intersects(b)]
            if intersecting:
                area_sqdeg = unary_union(intersecting).area
        area_m2 = round(area_sqdeg * sqrdeg_to_sqm, 2)
        f["properties"]["danger_area_m2"] = area_m2

        was_compliant = f["properties"]["compliant"]
        now_compliant = area_m2 < min_danger_area_m2
        f["properties"]["compliant"] = now_compliant
        if was_compliant and not now_compliant:
            f["properties"]["reclassified"] = "promoted_to_violation"
            n_promoted += 1
        elif (not was_compliant) and now_compliant:
            f["properties"]["reclassified"] = "demoted_to_compliant"
            n_demoted += 1

    print(f"danger-mass reclassification: "
          f"+{n_promoted} violations, -{n_demoted} demoted to compliant "
          f"(threshold {min_danger_area_m2} m^2)")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alerts-dir", type=Path, required=True,
                        help="folder with per-tile JSONs from compute_alerts")
    parser.add_argument("--tiles-root", type=Path,
                        default=Path("data/interim/tiles"),
                        help="root of GeoTIFF tiles, containing {sector}/*.tif")
    parser.add_argument("--output", type=Path, required=True,
                        help="output .geojson path")
    return parser.parse_args()


def _load_per_tile_features(json_files: list[Path], tiles_root: Path,
                            ) -> tuple[list[dict], float, int, int]:
    """Convert per-tile JSONs into georeferenced GeoJSON features.

    Returns ``(features, threshold_m, skipped_no_tif, skipped_no_polygon)``.
    ``threshold_m`` is taken from the first JSON that declares it.
    """
    features: list[dict] = []
    skipped_no_tif = 0
    skipped_no_polygon = 0
    threshold_m = DEFAULT_THRESHOLD_M
    sector_names = sorted(p.name for p in tiles_root.iterdir() if p.is_dir())

    for jf in json_files:
        data = json.loads(jf.read_text())
        if "threshold_m" in data:
            threshold_m = data["threshold_m"]
        tile_stem = Path(data["tile"]).stem
        tif = resolve_tif_path(tile_stem, tiles_root, sector_names)
        if tif is None:
            skipped_no_tif += 1
            continue
        with rasterio.open(tif) as src:
            transform = src.transform
            src_crs = src.crs
        model = data.get("model", "unknown")
        for b in data.get("buildings", []):
            feat = build_feature(b, tile_stem, model, transform, src_crs)
            if feat is None:
                skipped_no_polygon += 1
                continue
            features.append(feat)
            # Danger zones exist only for violations; each becomes its own feature.
            features.extend(build_danger_zone_features(
                b, tile_stem, model, transform, src_crs,
            ))
        for t in data.get("trees", []):
            feat = build_tree_feature(t, tile_stem, model, transform, src_crs)
            if feat is not None:
                features.append(feat)
    return features, threshold_m, skipped_no_tif, skipped_no_polygon


def main() -> None:
    args = _parse_args()

    json_files = [p for p in sorted(args.alerts_dir.glob("*.json"))
                  if p.stem != "summary"]

    features, threshold_m, skipped_no_tif, skipped_no_polygon = \
        _load_per_tile_features(json_files, args.tiles_root)

    print(f"raw features: {len(features)}")
    dedup_features = deduplicate(features, threshold_m=threshold_m)
    print(f"after dedup: {len(dedup_features)}")

    reclassify_by_danger_volume(dedup_features)

    # GeoJSON RFC 7946 requires WGS84 and omits a CRS member.
    fc = {"type": "FeatureCollection", "features": dedup_features}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fc, indent=2, allow_nan=False))
    print(f"wrote {len(dedup_features)} features to {args.output}")
    print(f"skipped: {skipped_no_tif} tiles without matching TIF, "
          f"{skipped_no_polygon} buildings without valid polygon")


if __name__ == "__main__":
    main()
