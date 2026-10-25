"""
modules/recommendation_engine.py
════════════════════════════════════════════════════════════════════════════════
Style-driven recommendation engine.
Each style (Vintage Indian, eclectic, mid_century_modern etc.) has:
  - Its own colour palette (wall/accent/ceiling)
  - Its own wallpaper patterns
  - Its own decor items
  - Its own budget distribution
  - Its own image generation prompt keywords

Changing the style chip NOW produces completely different output.
════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from modules.catalog_loader import WALLPAPER_CATALOGUE, DECOR_CATALOGUE

try:
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  STYLE DNA — every style has its own colour palette, feel, and budget weights
# ══════════════════════════════════════════════════════════════════════════════

STYLE_DNA = {
    "vintage_indian": {
        "palette": {
            "wall":    "#C4803A",   # deep ochre / turmeric
            "accent":  "#6B2737",   # deep maroon / sindoor red
            "ceiling": "#F5E6C8",   # aged parchment
            "floor":   "#8B5E3C",   # dark teak wood
            "note":    "Earthy ochres, deep maroons, and aged parchment — evoking haveli warmth.",
        },
        "colour_mood":    "terracotta, ochre, deep maroon, indigo, saffron, aged brass",
        "wallpaper_tags": ["jaali", "mughal", "kalamkari", "block print", "indigo", "lotus", "ancient"],
        "decor_tags":     ["murti", "wooden swing", "brass diya", "jharokha", "handloom", "terracotta pot", "carved wood"],
        "budget_weights": {"wallpaper":0.20,"furniture":0.30,"curtains":0.10,"rugs":0.15,"lamps":0.08,"wall_art":0.10,"accent":0.07},
    },
    "contemporary_indian": {
        "palette": {
            "wall":    "#E8D5A3",   # warm linen
            "accent":  "#B87333",   # copper
            "ceiling": "#FAF7F0",   # clean white
            "floor":   "#A0784A",   # medium wood
            "note":    "Modern Indian sensibility — clean lines, warm neutrals with brass and copper accents.",
        },
        "colour_mood":    "warm linen, copper, brass, terracotta, sage, clean white",
        "wallpaper_tags": ["geometric", "contemporary", "block print", "abstract", "brass", "minimal indian"],
        "decor_tags":     ["brass vase", "block print cushion", "sheesham table", "copper lamp", "cane chair", "dhurrie"],
        "budget_weights": {"wallpaper":0.15,"furniture":0.38,"curtains":0.08,"rugs":0.12,"lamps":0.10,"wall_art":0.08,"accent":0.09},
    },
    "scandinavian": {
        "palette": {
            "wall":    "#E8E2D5",   # warm off-white
            "accent":  "#7A9E7E",   # sage green
            "ceiling": "#F5F2EC",   # pure off-white
            "floor":   "#C4A882",   # pale oak
            "note":    "Wabi-sabi meets Scandinavian calm — muted, minimal, purposeful.",
        },
        "colour_mood":    "warm off-white, sage green, pale oak, charcoal, linen",
        "wallpaper_tags": ["minimal", "linen texture", "plain", "subtle stripe", "natural", "wabi-sabi"],
        "decor_tags":     ["linen sofa", "rattan lamp", "ceramic vase", "jute rug", "wooden shelf", "bonsai"],
        "budget_weights": {"wallpaper":0.12,"furniture":0.42,"curtains":0.08,"rugs":0.12,"lamps":0.10,"wall_art":0.08,"accent":0.08},
    },
    "eclectic": {
        "palette": {
            "wall":    "#D4956A",   # warm terracotta orange
            "accent":  "#6B4E8A",   # jewel purple
            "ceiling": "#F0E6D3",   # warm cream
            "floor":   "#8B5E3C",   # dark wood
            "note":    "Layered jewel tones, global textiles, maximalist warmth.",
        },
        "colour_mood":    "terracotta, jewel purple, teal, saffron, magenta, warm cream",
        "wallpaper_tags": ["floral", "botanical", "maximalist", "pattern", "eclectic", "global", "mandala"],
        "decor_tags":     ["macrame", "rattan chair", "kilim rug", "pendant lamp", "wall tapestry", "plants", "colourful cushions"],
        "budget_weights": {"wallpaper":0.18,"furniture":0.28,"curtains":0.10,"rugs":0.15,"lamps":0.08,"wall_art":0.12,"accent":0.09},
    },
    "mid_century_modern": {
        "palette": {
            "wall":    "#D4A843",   # mustard yellow
            "accent":  "#8B4513",   # burnt sienna
            "ceiling": "#F5F0E0",   # cream
            "floor":   "#7A5230",   # walnut
            "note":    "Retro optimism — warm wood tones, mustard, burnt orange, clean lines.",
        },
        "colour_mood":    "mustard yellow, burnt orange, walnut brown, avocado green, cream",
        "wallpaper_tags": ["geometric", "retro", "atomic", "stripe", "mid century", "sunburst"],
        "decor_tags":     ["walnut coffee table", "tulip chair", "arc floor lamp", "geometric rug", "abstract art", "credenza"],
        "budget_weights": {"wallpaper":0.15,"furniture":0.42,"curtains":0.07,"rugs":0.12,"lamps":0.10,"wall_art":0.08,"accent":0.06},
    },
    "coastal": {
        "palette": {
            "wall":    "#D4EAF0",   # sea glass blue
            "accent":  "#8FA88B",   # sea grass green
            "ceiling": "#FFFFFF",   # bright white
            "floor":   "#D4C5A0",   # driftwood
            "note":    "Bleached whites, sea blues, rattan — breezy and light.",
        },
        "colour_mood":    "sea glass blue, white, driftwood, coral, sea grass",
        "wallpaper_tags": ["stripe", "nautical", "seagrass", "linen", "coastal", "white", "shell"],
        "decor_tags":     ["rattan sofa", "white curtains", "jute rug", "coral accent", "driftwood shelf", "sea glass vase"],
        "budget_weights": {"wallpaper":0.12,"furniture":0.38,"curtains":0.12,"rugs":0.12,"lamps":0.10,"wall_art":0.08,"accent":0.08},
    },
    "victorian": {
        "palette": {
            "wall":    "#4A3728",   # dark walnut brown
            "accent":  "#8B1A2F",   # deep burgundy
            "ceiling": "#3A2A1A",   # near-black brown
            "floor":   "#2C1810",   # dark mahogany
            "note":    "Ink, mahogany, candlelight — moody intellectualism meets library aesthetics.",
        },
        "colour_mood":    "dark walnut, deep burgundy, aged gold, forest green, near-black",
        "wallpaper_tags": ["damask", "dark", "moody", "botanical", "library", "tartan", "vintage print"],
        "decor_tags":     ["leather armchair", "mahogany shelf", "globe lamp", "Persian rug", "antique map", "candlestick"],
        "budget_weights": {"wallpaper":0.18,"furniture":0.38,"curtains":0.08,"rugs":0.15,"lamps":0.10,"wall_art":0.08,"accent":0.03},
    },
    "shabby_chic_style": {
        "palette": {
            "wall":    "#C8D8B0",   # sage green
            "accent":  "#E8B4A0",   # blush pink
            "ceiling": "#F5F0E8",   # warm cream
            "floor":   "#C4A882",   # honey oak
            "note":    "Floral prints, sage greens, buttery creams — pastoral romanticism.",
        },
        "colour_mood":    "sage green, blush pink, buttercup yellow, lavender, warm cream",
        "wallpaper_tags": ["floral", "ditsy print", "botanical", "gingham", "toile", "vintage floral", "rose"],
        "decor_tags":     ["floral cushions", "rattan armchair", "dried flower wreath", "china plates", "lace curtains", "herb pots"],
        "budget_weights": {"wallpaper":0.20,"furniture":0.30,"curtains":0.12,"rugs":0.10,"lamps":0.10,"wall_art":0.10,"accent":0.08},
    },
    "industrial": {
        "palette": {
            "wall":    "#6B6560",   # concrete grey
            "accent":  "#B87333",   # copper
            "ceiling": "#4A4540",   # dark steel
            "floor":   "#2C2825",   # dark concrete
            "note":    "Raw concrete, exposed steel, charcoal — urban loft energy.",
        },
        "colour_mood":    "concrete grey, copper, steel, charcoal, matte black",
        "wallpaper_tags": ["brick", "concrete", "metal", "geometric", "dark", "exposed", "minimal"],
        "decor_tags":     ["steel shelf", "Edison bulb lamp", "leather sofa", "metal stool", "concrete planter", "geometric rug"],
        "budget_weights": {"wallpaper":0.12,"furniture":0.45,"curtains":0.05,"rugs":0.10,"lamps":0.15,"wall_art":0.08,"accent":0.05},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  CATALOGUE — real product data, loaded from catalog_seed.csv via catalog_loader.py
#  (WALLPAPER_CATALOGUE / DECOR_CATALOGUE imported at the top of this file)
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  STYLE-DRIVEN PALETTE OVERRIDE
# ══════════════════════════════════════════════════════════════════════════════

def get_style_palette(styles: list[str], dominant_colors: list[dict], lighting: str) -> dict:
    """
    Returns wall/accent/ceiling colours driven PRIMARILY by selected style.
    Blends with extracted dominant colour for personalisation.
    """
    # Primary style drives the palette
    primary = styles[0] if styles else "scandinavian"
    dna     = STYLE_DNA.get(primary, STYLE_DNA["scandinavian"])
    pal     = dna["palette"]

    # Slight blend with dominant extracted colour (10% influence)
    wall_hex = pal["wall"]
    if dominant_colors:
        dom_rgb  = dominant_colors[0]["rgb"]
        w_rgb    = [int(wall_hex[i:i+2],16) for i in (1,3,5)]
        blended  = [int(w*0.90 + d*0.10) for w, d in zip(w_rgb, dom_rgb)]
        wall_hex = "#{:02X}{:02X}{:02X}".format(*blended)

    # Dim lighting pushes wall slightly warmer
    if lighting == "dim":
        r, g, b  = [int(wall_hex[i:i+2],16) for i in (1,3,5)]
        wall_hex = "#{:02X}{:02X}{:02X}".format(min(255,r+10), g, max(0,b-8))

    return {
        "wall_color":          wall_hex,
        "accent_color":        pal["accent"],
        "ceiling_color":       pal["ceiling"],
        "object_driven":       True,
        "anchor_object":       None,
        "anchor_color":        dominant_colors[0]["hex"] if dominant_colors else "#C4A882",
        "palette_strategy":    "style_driven",
        "object_palette_note": pal["note"],
        "detected_style":      primary,
        "style_similarities":  {s: 1.0 if s == primary else 0.3 for s in styles},
        "colour_mood":         dna["colour_mood"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BUDGET ALLOCATION — per style
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_BUDGET_WEIGHTS = {
    "wallpaper":0.15,"furniture":0.38,"curtains":0.08,
    "rugs":0.12,"lamps":0.10,"wall_art":0.09,"accent":0.08
}

def allocate_budget(total_budget: float, scope: str, styles: list[str]) -> dict[str, float]:
    primary = styles[0] if styles else "scandinavian"
    weights = STYLE_DNA.get(primary, {}).get("budget_weights", DEFAULT_BUDGET_WEIGHTS).copy()

    # Accepts both the original scope vocabulary ("decor"/"interior"/"both")
    # and the frontend's design-path vocabulary ("home_decor"/
    # "interior_design"/"full_makeover") — normalise to the original three
    # before branching, so either caller works without duplicating logic.
    SCOPE_ALIASES = {
        "home_decor": "decor",
        "interior_design": "interior",
        "full_makeover": "both",
    }
    scope = SCOPE_ALIASES.get(scope, scope)

    if scope == "decor":
        furniture_w = weights.pop("furniture", 0.38)
        for k in weights:
            weights[k] += furniture_w / len(weights)
    elif scope == "interior":
        weights["furniture"] = 0.55
        weights["wallpaper"] = 0.20
        rest = (1.0 - 0.55 - 0.20) / max(len(weights) - 2, 1)
        for k in weights:
            if k not in ("furniture", "wallpaper"):
                weights[k] = rest
    # "both" / anything else (including full_makeover): keep the style's
    # own natural budget_weights unchanged — full makeover means every
    # category (furniture + wallpaper + everything) gets its normal share.

    # Normalise to sum = 1
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    return {cat: round(total_budget * w, 2) for cat, w in weights.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _style_match(item_styles: list[str], user_styles: list[str]) -> float:
    """1.0 if exact match, 0.5 if partial, 0.05 if none — makes style decisive."""
    if not user_styles:
        return 0.5
    exact   = len(set(item_styles) & set(user_styles))
    if exact == 0:
        return 0.05   # near-zero so off-style items almost never win
    return min(1.0, exact / len(user_styles) * 1.2)


def _colour_compat(item_rgb: list[int], dominant: list[dict]) -> float:
    if not dominant:
        return 0.5
    dom_rgb = dominant[0]["rgb"]
    dist    = sum((a-b)**2 for a,b in zip(item_rgb, dom_rgb)) ** 0.5
    return round(max(0.0, 1.0 - dist/441.7), 4)


def _score(item_styles, item_rgb, user_styles, dominant_colors):
    """Style is 80% of the score — guarantees different styles = different results."""
    return 0.80 * _style_match(item_styles, user_styles) + \
           0.20 * _colour_compat(item_rgb, dominant_colors)


# ══════════════════════════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WallpaperRecommendation:
    id:str; name:str; style:list; scale:str; base_color:str
    price_sqft:float; pattern_type:str; similarity:float
    budget_label:str; estimated_total:float

@dataclass
class DecorRecommendation:
    id:str; name:str; category:str; style:list; color:str
    price:float; similarity:float; colour_score:float
    budget_label:str; budget_pct:float

@dataclass
class RecommendationResult:
    wallpapers:list; decor_items:list; budget_breakdown:dict
    total_estimated:float; budget_remaining:float; summary:str


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RECOMMEND
# ══════════════════════════════════════════════════════════════════════════════

def recommend(
    dominant_colors:  list[dict],
    room_type:        str,
    size_category:    str,
    styles:           list[str],
    total_budget:     float,
    scope:            str,
    wall_area_sqft:   float = 200.0,
    top_n_wallpaper:  int   = 6,
    top_n_decor:      int   = 8,
    image_features=None,
    object_colors:    list  = None,
    style_sims:       dict  = None,
) -> RecommendationResult:

    object_colors = object_colors or []
    budget_map    = allocate_budget(total_budget, scope, styles)

    # ── Preferred scale by room size ─────────────────────────────────────────
    preferred_scales = {"small":["plain","small"],"medium":["small","medium"],"large":["large","medium"]}.get(size_category, ["medium"])

    # ── WALLPAPERS ────────────────────────────────────────────────────────────
    wp_budget_sqft = budget_map.get("wallpaper", 0) / max(wall_area_sqft, 1)
    # Filter by style first, relax if too few
    filtered_wp = [w for w in WALLPAPER_CATALOGUE if any(s in styles for s in w["style"])]
    if len(filtered_wp) < 3:
        filtered_wp = WALLPAPER_CATALOGUE   # open up if no matches

    wp_scores = []
    for w in filtered_wp:
        s = _score(w["style"], w["base_color"], styles, dominant_colors)
        s += 0.05 if w["scale"] in preferred_scales else 0.0
        s += 0.03 if room_type in w.get("suitable_rooms", []) else 0.0
        wp_scores.append((w, s))

    wp_scores.sort(key=lambda x: x[1], reverse=True)
    wallpapers = []
    for w, score in wp_scores[:top_n_wallpaper]:
        est     = round(w["price_sqft"] * wall_area_sqft, 2)
        r,g,b   = w["base_color"]
        wallpapers.append(WallpaperRecommendation(
            id=w["id"], name=w["name"], style=w["style"], scale=w["scale"],
            base_color=f"#{r:02X}{g:02X}{b:02X}", price_sqft=w["price_sqft"],
            pattern_type=w["pattern_type"], similarity=round(min(score,0.99),4),
            budget_label="In Budget" if w["price_sqft"]<=wp_budget_sqft else "Over Budget",
            estimated_total=est,
        ))

    # ── DÉCOR ─────────────────────────────────────────────────────────────────
    decor_items = []
    for cat, cat_budget in budget_map.items():
        if cat == "wallpaper":
            continue

        # Filter by category + style, relax if empty
        cat_items = [d for d in DECOR_CATALOGUE
                     if d["category"]==cat and any(s in styles for s in d["style"])
                     and d["price"] <= cat_budget * 1.5]
        if not cat_items:
            cat_items = [d for d in DECOR_CATALOGUE
                         if d["category"]==cat and d["price"] <= cat_budget * 1.5]
        if not cat_items:
            continue

        scored = []
        for d in cat_items:
            s       = _score(d["style"], d["color"], styles, dominant_colors)
            c_score = _colour_compat(d["color"], dominant_colors)
            scored.append((d, s, c_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        d, score, c_score = scored[0]
        r,g,b = d["color"]
        decor_items.append(DecorRecommendation(
            id=d["id"], name=d["name"], category=d["category"],
            style=d["style"], color=f"#{r:02X}{g:02X}{b:02X}",
            price=d["price"], similarity=round(min(score,0.99),4),
            colour_score=round(c_score,4),
            budget_label="In Budget" if d["price"]<=cat_budget else "Over Budget",
            budget_pct=round(d["price"]/max(cat_budget,1)*100,1),
        ))

    decor_items.sort(key=lambda x: x.similarity, reverse=True)
    decor_items = decor_items[:top_n_decor]

    # ── TOTALS ────────────────────────────────────────────────────────────────
    wall_total  = wallpapers[0].estimated_total if wallpapers else 0
    decor_total = sum(d.price for d in decor_items)
    total_est   = round(wall_total + decor_total, 2)
    remaining   = round(total_budget - total_est, 2)

    primary_style = styles[0] if styles else "scandinavian"
    summary = (
        f"{primary_style} recommendations for your {size_category} {room_type.replace('_',' ')}: "
        f"{len(wallpapers)} wallpapers, {len(decor_items)} décor items curated. "
        f"Total estimate ₹{total_est:,.0f} "
        f"({'saving ₹'+f'{remaining:,.0f}' if remaining>=0 else 'over by ₹'+f'{abs(remaining):,.0f}'})."
    )

    return RecommendationResult(
        wallpapers=wallpapers, decor_items=decor_items,
        budget_breakdown=budget_map,
        total_estimated=total_est, budget_remaining=remaining,
        summary=summary,
    )


def recommendation_to_dict(result: RecommendationResult) -> dict:
    return {
        "wallpapers": [{"id":w.id,"name":w.name,"style":w.style,"scale":w.scale,"base_color":w.base_color,"price_sqft":w.price_sqft,"pattern_type":w.pattern_type,"similarity_score":w.similarity,"budget_label":w.budget_label,"estimated_total":w.estimated_total} for w in result.wallpapers],
        "decor_items":[{"id":d.id,"name":d.name,"category":d.category,"style":d.style,"color":d.color,"price":d.price,"similarity_score":d.similarity,"colour_score":d.colour_score,"budget_label":d.budget_label,"budget_pct":d.budget_pct} for d in result.decor_items],
        "budget_breakdown":result.budget_breakdown,
        "total_estimated":result.total_estimated,
        "budget_remaining":result.budget_remaining,
        "summary":result.summary,
    }
