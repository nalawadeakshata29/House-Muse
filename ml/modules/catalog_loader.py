"""
modules/catalog_loader.py
════════════════════════════════════════════════════════════════════════════
Loads the real product catalog (data/catalog_seed.csv) and reshapes it into
the exact structures recommendation_engine.py expects for WALLPAPER_CATALOGUE
and DECOR_CATALOGUE — replacing the small hand-written placeholder lists
that used to live directly in that file, so real product names/prices
actually drive recommendations and the budget breakdown.

STYLE MAPPING
────────────────────────────────────────────────────────────────────────────
catalog_seed.csv's style_tags map onto this app's internal style ids (the
keys of STYLE_DNA in recommendation_engine.py) via STYLE_TAG_TO_ID below.
Most map directly by name. Two are worth knowing about:

    Japandi  → scandinavian       (minimal, linen, wabi-sabi — same mood)
    Bohemian → eclectic           (maximalist, global, layered — same mood)

Everything else (Vintage Indian, Contemporary Indian, Coastal, Industrial,
Victorian, Mid-Century Modern, Cottagecore→shabby_chic_style) has catalog
rows written specifically for that internal style, so no guessing involved.
"""
import csv
import os

_CSV_PATH = os.environ.get(
    "CATALOG_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "catalog_seed.csv"),
)
if not os.path.exists(_CSV_PATH):
    _backend_csv = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "data", "catalog_seed.csv")
    if os.path.exists(_backend_csv):
        _CSV_PATH = _backend_csv

STYLE_TAG_TO_ID = {
    "Vintage Indian": "vintage_indian",
    "Contemporary Indian": "contemporary_indian",
    "Coastal": "coastal",
    "Industrial": "industrial",
    "Japandi": "scandinavian",
    "Bohemian": "eclectic",
    "Victorian": "victorian",
    "Mid-Century Modern": "mid_century_modern",
    "Cottagecore": "shabby_chic_style",
    "vintage_indian": "vintage_indian",
    "contemporary_indian": "contemporary_indian",
    "coastal": "coastal",
    "industrial": "industrial",
    "scandinavian": "scandinavian",
    "eclectic": "eclectic",
    "victorian": "victorian",
    "mid_century_modern": "mid_century_modern",
    "shabby_chic_style": "shabby_chic_style",
}

# Maps the CSV's decor_item categories onto the 7 budget buckets
# recommendation_engine.py's STYLE_DNA["*"]["budget_weights"] uses.
CATEGORY_TO_BUDGET_BUCKET = {
    "curtain":  "curtains",
    "cushion":  "accent",
    "lighting": "lamps",
    "mirror":   "accent",
    "plant":    "accent",
    "rug":      "rugs",
    "seating":  "furniture",
    "storage":  "furniture",
    "table":    "furniture",
    "wall_art": "wall_art",
    # "paint" and "wallpaper" are item_type="wall_option", handled separately
    # below — they don't go through this map.
}

# Used only when a CSV row's hex_color is blank — most wallpaper rows don't
# have one (pattern name only), paint rows almost always do.
_DEFAULT_WALLPAPER_COLOR = [200, 190, 175]
_DEFAULT_DECOR_COLOR = [170, 150, 130]


def _hex_to_rgb(hex_str: str, fallback: list[int]) -> list[int]:
    hex_str = (hex_str or "").strip().lstrip("#")
    if len(hex_str) != 6:
        return fallback
    try:
        return [int(hex_str[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return fallback


def _load_rows() -> list[dict]:
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build() -> tuple[list[dict], list[dict]]:
    rows = _load_rows()
    wallpapers, decor = [], []
    wp_n, decor_n = 0, 0

    for row in rows:
        style_id = STYLE_TAG_TO_ID.get((row.get("style_tags") or "").strip())
        if not style_id:
            continue  # unmapped style tag (currently just "Art Deco") — skip

        try:
            price = float(row["price"])
        except (KeyError, ValueError):
            continue
        name = row.get("name", "").strip()
        if not name:
            continue

        if row.get("item_type") == "wall_option":
            wp_n += 1
            category = row.get("category", "")
            wallpapers.append({
                "id": f"CAT-WP-{wp_n:03d}",
                "name": name,
                "style": [style_id],
                "scale": "medium",  # not present in the CSV — neutral default
                "base_color": _hex_to_rgb(row.get("hex_color"), _DEFAULT_WALLPAPER_COLOR),
                "price_sqft": price,
                "pattern_type": row.get("pattern") or ("paint" if category == "paint" else "plain"),
                "suitable_rooms": [],  # unknown per-row — no bonus, not a hard filter
                "budget_tier": row.get("budget_tier", ""),
            })
        else:
            bucket = CATEGORY_TO_BUDGET_BUCKET.get(row.get("category", ""))
            if not bucket:
                continue
            decor_n += 1
            decor.append({
                "id": f"CAT-DE-{decor_n:03d}",
                "name": name,
                "category": bucket,
                "style": [style_id],
                "color": _hex_to_rgb(row.get("hex_color"), _DEFAULT_DECOR_COLOR),
                "price": price,
                "room": [],
                "budget_tier": row.get("budget_tier", ""),
            })

    return wallpapers, decor


try:
    WALLPAPER_CATALOGUE, DECOR_CATALOGUE = _build()
except FileNotFoundError:
    # Lets the service still start (e.g. under a test runner) even if the
    # CSV isn't deployed alongside this module yet — recommend() already
    # handles empty catalogues by simply returning fewer results.
    WALLPAPER_CATALOGUE, DECOR_CATALOGUE = [], []
