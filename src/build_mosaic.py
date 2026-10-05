"""Build a downsampled PNG mosaic of a sector's GeoTIFF tiles.

Merges all ``*.tif`` tiles under ``data/interim/tiles/<sector>/`` into a
single raster in Web Mercator, downsamples with PIL for file size, and
saves it as a PNG plus a sidecar JSON with WGS84 bounds ready for use as
a Folium ImageOverlay.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.warp import transform as warp_transform

# Reproject to Web Mercator so the raster is axis-aligned in the same
# projection Leaflet renders in. The sidecar's WGS84 bounds come from
# projecting the raster's two Web Mercator corners back to lat/lon;
# Leaflet reprojects them forward and recovers the exact Web Mercator
# rectangle, so the overlay lines up pixel-for-pixel with the base tiles.
DST_CRS = "EPSG:3857"
WGS84 = "EPSG:4326"

# Default max side of the output PNG. Larger values keep more detail at
# high zoom but the resulting HTML grows quickly because Folium embeds
# the image as base64.
DEFAULT_MAX_DIM_PX = 3000

# Expected number of bands in the source tiles (RGBA).
EXPECTED_BANDS = 4


# ---------------------------------------------------------------------------
# merge, reproject, encode
# ---------------------------------------------------------------------------

def _merge_native(tif_paths: list[Path]):
    """Merge all TIF tiles at native resolution in the source UTM CRS.

    ``rasterio.merge`` aloca el bounding box completo que contiene a todos
    los tiles, lo cual puede ser muy grande (varios GB) cuando los tiles
    estan dispersos geograficamente. Para sectores con mas de ~20 tiles
    separados, considerar llamar este modulo por sub-sector y combinar los
    mosaicos resultantes a nivel PNG, o usar ``bounds`` explicito para
    recortar el area de interes antes del merge.
    """
    srcs = [rasterio.open(p) for p in tif_paths]
    try:
        mosaic, out_transform = merge(srcs)
        return mosaic, out_transform, srcs[0].crs
    finally:
        for s in srcs:
            s.close()


def _reproject_to_webmercator(mosaic: np.ndarray, out_transform,
                              src_crs) -> tuple[np.ndarray, object]:
    """Warp ``mosaic`` from ``src_crs`` to Web Mercator (returns array + transform)."""
    _, rows, cols = mosaic.shape
    west, south, east, north = array_bounds(rows, cols, out_transform)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, DST_CRS, cols, rows, west, south, east, north,
    )
    dst = np.zeros((mosaic.shape[0], dst_h, dst_w), dtype=mosaic.dtype)
    for band in range(mosaic.shape[0]):
        reproject(
            source=mosaic[band],
            destination=dst[band],
            src_transform=out_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=DST_CRS,
            resampling=Resampling.bilinear,
        )
    return dst, dst_transform


def _wgs84_bounds_of(raster: np.ndarray, transform) -> tuple[float, float, float, float]:
    """Return ``(lon_min, lat_min, lon_max, lat_max)`` for a Web Mercator raster."""
    _, rows, cols = raster.shape
    merc_w, merc_s, merc_e, merc_n = array_bounds(rows, cols, transform)
    lons, lats = warp_transform(DST_CRS, WGS84,
                                [merc_w, merc_e], [merc_s, merc_n])
    return lons[0], lats[0], lons[1], lats[1]


def _to_rgba_image(reprojected: np.ndarray) -> Image.Image:
    """Convert a 4-band UInt8 array (H, W, 4) to a PIL RGBA image.

    The alpha channel comes from the source TIF; we do not synthesize one
    from "black pixel means padding" because lake water and shadows can
    genuinely be near zero and would end up transparent.
    """
    if reprojected.shape[0] != EXPECTED_BANDS:
        raise SystemExit(
            "expected 4-band RGBA tiles; source has no alpha, cannot "
            "distinguish reprojection padding from real dark pixels."
        )
    rgb = np.transpose(reprojected[:3], (1, 2, 0))
    alpha = reprojected[3].astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


def _uniform_downsample(img: Image.Image, max_dim_px: int) -> Image.Image:
    """Resize so that the longer side is at most ``max_dim_px``."""
    w, h = img.size
    scale = min(1.0, max_dim_px / max(w, h))
    if scale >= 1.0:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _write_bounds_sidecar(out_bounds: Path, out_png: Path, img: Image.Image,
                          native_rows: int, native_cols: int,
                          bounds: tuple[float, float, float, float],
                          src_crs, n_tiles: int) -> None:
    """Write the JSON sidecar consumed by ``generate_map``."""
    lon_min, lat_min, lon_max, lat_max = bounds
    out_bounds.write_text(json.dumps({
        "png": out_png.name,
        "bounds_wgs84": {
            "south": lat_min,
            "west": lon_min,
            "north": lat_max,
            "east": lon_max,
        },
        "size_native": {"rows": native_rows, "cols": native_cols},
        "size_out": {"rows": img.size[1], "cols": img.size[0]},
        "source_crs": src_crs.to_string(),
        "n_tiles_merged": n_tiles,
    }, indent=2))


def build_mosaic(sector_dir: Path, max_dim_px: int, out_png: Path,
                 out_bounds: Path) -> None:
    """Full pipeline: merge → reproject to WebMercator → downsample → save."""
    tif_paths = sorted(sector_dir.glob("*.tif"))
    if not tif_paths:
        raise SystemExit(f"no tiles in {sector_dir}")

    mosaic, out_transform, src_crs = _merge_native(tif_paths)
    _, native_rows, native_cols = mosaic.shape

    reprojected, dst_transform = _reproject_to_webmercator(
        mosaic, out_transform, src_crs
    )
    bounds = _wgs84_bounds_of(reprojected, dst_transform)

    img = _uniform_downsample(_to_rgba_image(reprojected), max_dim_px)

    # PNG keeps the alpha channel; JPG would lose it and reintroduce black.
    if out_png.suffix.lower() != ".png":
        out_png = out_png.with_suffix(".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, optimize=True)

    _write_bounds_sidecar(out_bounds, out_png, img,
                          native_rows, native_cols, bounds,
                          src_crs, len(tif_paths))
    print(f"wrote {out_png} ({img.size[0]}x{img.size[1]} px) and {out_bounds}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector-dir", type=Path, required=True,
                        help="folder with source .tif tiles")
    parser.add_argument("--max-dim-px", type=int, default=DEFAULT_MAX_DIM_PX,
                        help="downscale so the longer side has at most this "
                             "many pixels (uniform scale, no snap)")
    parser.add_argument("--output", type=Path, required=True,
                        help="output PNG path (sidecar .json is written next to it)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_bounds = args.output.with_suffix(".json")
    build_mosaic(args.sector_dir, args.max_dim_px, args.output, out_bounds)


if __name__ == "__main__":
    main()
