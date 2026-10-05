"""Render a per-sector alert GeoJSON as an interactive HTML map.

Uses Folium (Leaflet.js wrapper) with two base layers (Esri Street default,
Esri Satellite optional via ``BASE_LAYER=satellite``). Each building is drawn
as a polygon colored by compliance status, with a popup showing the minimum
distance to vegetation and metadata.
"""

import argparse
import json
import os
from pathlib import Path

import folium

# Colors. Blue = compliant, red = violation, yellow = danger-zone vegetation,
# dark green = detected tree instance.
COLOR_OK = "#2e86de"
COLOR_VIOLATION = "#e84118"
COLOR_DANGER_ZONE = "#ffcc00"
COLOR_TREE = "#2e7d32"

# Map defaults.
INITIAL_ZOOM = 17
MAX_ZOOM = 22

# Base tiles: no API key, no file:// blocking. Street is the default; a
# vector-streets layer gives useful context for the alerts even though in
# rural Villa La Angostura it can carry a multi-metre offset against the
# drone orthophoto. The satellite layer is kept as an option (``BASE_LAYER=
# satellite``) because its imagery lines up pixel-for-pixel.
ESRI_SATELLITE_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_SATELLITE_ATTR = "Tiles &copy; Esri &mdash; World Imagery"
ESRI_STREET_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
)
ESRI_STREET_ATTR = "Tiles &copy; Esri &mdash; OpenStreetMap contributors"

# Threshold used in the legend text (matches the value hardcoded in the
# rule below; kept as a constant here for easier edits).
LEGEND_THRESHOLD_M = 2


# ---------------------------------------------------------------------------
# GeoJSON styling and helpers
# ---------------------------------------------------------------------------

def style_feature(feature: dict) -> dict:
    """Return a Leaflet style dict for one GeoJSON feature."""
    props = feature["properties"]
    kind = props.get("kind")
    if kind == "danger_zone":
        return {
            "fillColor": COLOR_DANGER_ZONE,
            "color": COLOR_DANGER_ZONE,
            "weight": 1,
            "fillOpacity": 0.7,
        }
    if kind == "tree":
        return {
            "fillColor": COLOR_TREE,
            "color": COLOR_TREE,
            "weight": 0.5,
            "fillOpacity": 0.35,
        }
    fill = COLOR_OK if props["compliant"] else COLOR_VIOLATION
    return {
        "fillColor": fill,
        "color": fill,
        "weight": 1.5,
        "fillOpacity": 0.55,
    }


def compute_center(features: list[dict]) -> tuple[float, float]:
    """Mean (lat, lon) of every polygon ring vertex in the collection."""
    lons, lats = [], []
    for f in features:
        geom = f["geometry"]
        if geom["type"] == "Polygon":
            rings = [geom["coordinates"][0]]
        elif geom["type"] == "MultiPolygon":
            rings = [poly[0] for poly in geom["coordinates"]]
        else:
            continue
        for ring in rings:
            for x, y in ring:
                lons.append(x)
                lats.append(y)
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _keep_violation_or_danger(feature: dict) -> bool:
    """Filter for ``--only-violations``: keep non-compliant, at-edge, danger.

    Tree features no tienen ``compliant`` (son solo coberturas), así que
    se descartan directamente para no romper el filtro.
    """
    p = feature["properties"]
    kind = p.get("kind")
    if kind == "danger_zone":
        return True
    if kind == "tree":
        return False
    return (not p["compliant"]) or p.get("at_edge")


# ---------------------------------------------------------------------------
# map assembly
# ---------------------------------------------------------------------------

def _add_orthomosaic(m: folium.Map, orthomosaic: Path) -> None:
    """Overlay a georeferenced PNG mosaic (with sidecar JSON) on the map."""
    meta = json.loads(orthomosaic.with_suffix(".json").read_text())
    b = meta["bounds_wgs84"]
    folium.raster_layers.ImageOverlay(
        image=str(orthomosaic),
        bounds=[[b["south"], b["west"]], [b["north"], b["east"]]],
        opacity=1.0,
        name="Drone orthomosaic",
        interactive=False,
        cross_origin=False,
    ).add_to(m)


def _add_geojson_layer(m: folium.Map, fc: dict) -> None:
    """Add the styled GeoJSON layer with popup and tooltip."""
    folium.GeoJson(
        fc,
        name="Detecciones",
        style_function=style_feature,
        tooltip=folium.GeoJsonTooltip(
            fields=["building_id", "distance_m", "compliant"],
            aliases=["ID", "distance (m)", "compliant"],
        ),
        popup=folium.GeoJsonPopup(
            fields=["building_id", "distance_m", "compliant", "at_edge",
                    "tile_id", "model"],
            aliases=["ID", "distance (m)", "compliant", "at edge",
                     "tile", "model"],
        ),
    ).add_to(m)


def _slider_html(map_name: str) -> str:
    """Return an HTML+JS snippet with per-class toggles and confidence sliders.

    Edifices and vegetation can be toggled independently; each class has its
    own confidence threshold. Danger zones are tied to vegetation (same
    score) so they follow its toggle and slider.
    """
    return """
    <div id=\"conf-widget\" style=\"
        position: fixed; bottom: 30px; right: 30px;
        background: white; padding: 10px 14px; border: 1px solid #999;
        border-radius: 6px; font-family: sans-serif; font-size: 13px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15); z-index: 9999;
        min-width: 240px;\">
      <div style=\"font-weight: bold; margin-bottom: 6px;\">Capas y confianza</div>
      <div style=\"display:flex; align-items:center; gap:6px; margin-top:4px;\">
        <input type=\"checkbox\" id=\"show-buildings\" checked>
        <label for=\"show-buildings\" style=\"flex:1;\">Edificios</label>
        <span id=\"b-counter\" style=\"color:#555; font-variant-numeric: tabular-nums;\"></span>
      </div>
      <input id=\"b-slider\" type=\"range\" min=\"15\" max=\"95\" step=\"1\"
             value=\"45\" style=\"width: 100%;\">
      <div style=\"color:#555; font-variant-numeric: tabular-nums;\">
        <span id=\"b-value\">45 por ciento</span>
      </div>

      <div style=\"display:flex; align-items:center; gap:6px; margin-top:10px;\">
        <input type=\"checkbox\" id=\"show-trees\" checked>
        <label for=\"show-trees\" style=\"flex:1;\">Vegetacion</label>
        <span id=\"t-counter\" style=\"color:#555; font-variant-numeric: tabular-nums;\"></span>
      </div>
      <input id=\"t-slider\" type=\"range\" min=\"15\" max=\"95\" step=\"1\"
             value=\"45\" style=\"width: 100%;\">
      <div style=\"color:#555; font-variant-numeric: tabular-nums;\">
        <span id=\"t-value\">45 por ciento</span>
      </div>
    </div>
    <script>
    (function() {
      function setup() {
        var bSlider = document.getElementById('b-slider');
        var tSlider = document.getElementById('t-slider');
        var bLabel = document.getElementById('b-value');
        var tLabel = document.getElementById('t-value');
        var bCounter = document.getElementById('b-counter');
        var tCounter = document.getElementById('t-counter');
        var bToggle = document.getElementById('show-buildings');
        var tToggle = document.getElementById('show-trees');
        var mapObj = window.__MAP_NAME__;
        if (!mapObj) { setTimeout(setup, 100); return; }

        function effectiveViolation(p, tT) {
          // A building originally marked compliant stays compliant.
          // A building marked as violation is "really" a violation only if
          // at least one of the tree scores that triggered its danger zones
          // is still above the current vegetation threshold.
          if (p.compliant) return false;
          var scores = p.danger_tree_scores || [];
          if (scores.length === 0) return true;
          for (var i = 0; i < scores.length; i++) {
            if (scores[i] >= tT) return true;
          }
          return false;
        }
        function styleBuilding(layer, p, tT) {
          var violation = effectiveViolation(p, tT);
          var fill = violation ? '__COLOR_VIO__' : '__COLOR_OK__';
          layer.setStyle({color: fill, fillColor: fill,
                          weight: 1.5, fillOpacity: 0.55, opacity: 1});
        }
        function styleTree(layer) {
          layer.setStyle({color: '__COLOR_TREE__', fillColor: '__COLOR_TREE__',
                          weight: 0.5, fillOpacity: 0.35, opacity: 1});
        }
        function styleDanger(layer) {
          layer.setStyle({color: '__COLOR_DZ__', fillColor: '__COLOR_DZ__',
                          weight: 1, fillOpacity: 0.7, opacity: 1});
        }
        function hide(layer) {
          layer.setStyle({fillOpacity: 0, opacity: 0});
        }

        function apply() {
          var bT = parseFloat(bSlider.value) / 100.0;
          var tT = parseFloat(tSlider.value) / 100.0;
          bLabel.textContent = bSlider.value + ' por ciento';
          tLabel.textContent = tSlider.value + ' por ciento';
          var bShow = bToggle.checked, tShow = tToggle.checked;
          var bVis = 0, bTot = 0, tVis = 0, tTot = 0;
          mapObj.eachLayer(function(layer) {
            if (!layer.feature || !layer.feature.properties) return;
            var p = layer.feature.properties;
            var s = p.score;
            if (p.kind === 'building') {
              bTot++;
              var ok = bShow && (s == null || s >= bT);
              if (ok) { bVis++; styleBuilding(layer, p, tT); } else hide(layer);
            } else if (p.kind === 'tree') {
              tTot++;
              var ok = tShow && (s == null || s >= tT);
              if (ok) { tVis++; styleTree(layer); } else hide(layer);
            } else if (p.kind === 'danger_zone') {
              var ok = tShow && (s == null || s >= tT);
              if (ok) styleDanger(layer); else hide(layer);
            }
          });
          bCounter.textContent = bVis + ' / ' + bTot;
          tCounter.textContent = tVis + ' / ' + tTot;
        }
        bSlider.addEventListener('input', apply);
        tSlider.addEventListener('input', apply);
        bToggle.addEventListener('change', apply);
        tToggle.addEventListener('change', apply);
        apply();
      }
      window.addEventListener('load', setup);
    })();
    </script>
    """.replace("__MAP_NAME__", map_name) \
       .replace("__COLOR_OK__", COLOR_OK) \
       .replace("__COLOR_VIO__", COLOR_VIOLATION) \
       .replace("__COLOR_DZ__", COLOR_DANGER_ZONE) \
       .replace("__COLOR_TREE__", COLOR_TREE)


def _legend_html(title: str) -> str:
    """Return an HTML snippet for the fixed legend box (bottom-left)."""
    return f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px;
        background: white; padding: 10px 14px; border: 1px solid #999;
        border-radius: 6px; font-family: sans-serif; font-size: 13px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15); z-index: 9999;">
      <div style="font-weight: bold; margin-bottom: 6px;">{title}</div>
      <div><span style="background:{COLOR_OK}; display:inline-block;
                       width:12px; height:12px; margin-right:6px;"></span>
        Compliant (&gt;= {LEGEND_THRESHOLD_M} m)</div>
      <div><span style="background:{COLOR_VIOLATION}; display:inline-block;
                       width:12px; height:12px; margin-right:6px;"></span>
        Violation (&lt; {LEGEND_THRESHOLD_M} m)</div>
      <div><span style="background:{COLOR_DANGER_ZONE}; display:inline-block;
                       width:12px; height:12px; margin-right:6px;"></span>
        Nearby vegetation (&lt; {LEGEND_THRESHOLD_M} m)</div>
    </div>
    """


def _build_map(fc: dict, title: str, orthomosaic: Path | None) -> folium.Map:
    """Assemble the full Folium map for a filtered feature collection."""
    center_lat, center_lon = compute_center(fc["features"])
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=INITIAL_ZOOM,
        tiles=None,
        control_scale=True,
        max_zoom=MAX_ZOOM,
    )
    sat_show = os.environ.get("BASE_LAYER", "street").lower() == "satellite"
    folium.TileLayer(
        tiles=ESRI_SATELLITE_TILES,
        attr=ESRI_SATELLITE_ATTR,
        name="Satelite",
        max_zoom=MAX_ZOOM,
        overlay=False,
        control=True,
        show=sat_show,
    ).add_to(m)
    folium.TileLayer(
        tiles=ESRI_STREET_TILES,
        attr=ESRI_STREET_ATTR,
        name="Callejero (puede tener offset en zonas rurales)",
        max_zoom=MAX_ZOOM,
        overlay=False,
        control=True,
        show=not sat_show,
    ).add_to(m)
    if orthomosaic is not None:
        _add_orthomosaic(m, orthomosaic)
    _add_geojson_layer(m, fc)
    m.get_root().html.add_child(folium.Element(_legend_html(title)))
    m.get_root().html.add_child(folium.Element(_slider_html(m.get_name())))
    folium.LayerControl().add_to(m)
    return m


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="output .html path")
    parser.add_argument("--title", default="Defensible space alerts")
    parser.add_argument("--orthomosaic", type=Path,
                        help="optional PNG mosaic to show as base layer; "
                             "expects a sidecar .json with WGS84 bounds")
    parser.add_argument("--only-violations", action="store_true",
                        help="skip compliant buildings (keeps only violations "
                             "and at-edge cases)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fc = json.loads(args.geojson.read_text())

    if args.only_violations:
        before = len(fc["features"])
        fc["features"] = [f for f in fc["features"] if _keep_violation_or_danger(f)]
        print(f"filtered {before} -> {len(fc['features'])} features "
              f"(kept violations, at-edge and danger zones)")

    if not fc["features"]:
        print("no features to render")
        return

    m = _build_map(fc, args.title, args.orthomosaic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.output))
    print(f"wrote map with {len(fc['features'])} buildings to {args.output}")


if __name__ == "__main__":
    main()
