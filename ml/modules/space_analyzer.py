"""
space_analyzer.py
─────────────────
Module 3 — Space Analysis Logic
Inputs : room dimensions (L × W × H in feet)
Outputs: room size category, ceiling height class, area, volume, design rules
"""

from dataclasses import dataclass


# ─── Thresholds (in sq ft) ────────────────────────────────────────────────────
SMALL_MAX  = 120    # < 120 sq ft → small
MEDIUM_MAX = 300    # 120–300 sq ft → medium
# > 300 sq ft → large

CEILING_LOW      = 8.5   # ft
CEILING_STANDARD = 10.0  # ft
# > 10 ft → high


@dataclass
class SpaceProfile:
    length:         float
    width:          float
    height:         float
    area:           float
    volume:         float
    size_category:  str    # small | medium | large
    ceiling_class:  str    # low | standard | high
    aspect_ratio:   float  # length / width
    shape_hint:     str    # square | rectangular | elongated

    # Rule-based design guidance
    color_guidance:       str
    furniture_scale:      str
    pattern_scale:        str
    lighting_suggestions: list[str]
    layout_tips:          list[str]
    wallpaper_suitability: str


def analyse_space(
    length: float,
    width:  float,
    height: float,
    room_type: str = "living_room",
) -> SpaceProfile:
    """
    Derive design rules from raw room dimensions.

    Args:
        length    : room length in feet
        width     : room width in feet
        height    : ceiling height in feet
        room_type : affects specific layout and lighting advice

    Returns SpaceProfile dataclass.
    """
    area   = round(length * width,         2)
    volume = round(length * width * height, 2)

    # ── Size category ──
    if area < SMALL_MAX:
        size_cat = "small"
    elif area < MEDIUM_MAX:
        size_cat = "medium"
    else:
        size_cat = "large"

    # ── Ceiling height ──
    if height <= CEILING_LOW:
        ceil_class = "low"
    elif height <= CEILING_STANDARD:
        ceil_class = "standard"
    else:
        ceil_class = "high"

    # ── Aspect ratio / shape ──
    ratio = round(max(length, width) / max(min(length, width), 0.01), 2)
    if ratio < 1.2:
        shape = "square"
    elif ratio < 1.8:
        shape = "rectangular"
    else:
        shape = "elongated"

    # ── Generate design rules ──
    color_guidance       = _color_guidance(size_cat, ceil_class)
    furniture_scale      = _furniture_scale(size_cat)
    pattern_scale        = _pattern_scale(size_cat, shape)
    lighting_suggestions = _lighting_suggestions(size_cat, ceil_class, room_type)
    layout_tips          = _layout_tips(size_cat, shape, room_type)
    wallpaper_suit       = _wallpaper_suitability(size_cat, ceil_class, shape)

    return SpaceProfile(
        length=length,
        width=width,
        height=height,
        area=area,
        volume=volume,
        size_category=size_cat,
        ceiling_class=ceil_class,
        aspect_ratio=ratio,
        shape_hint=shape,
        color_guidance=color_guidance,
        furniture_scale=furniture_scale,
        pattern_scale=pattern_scale,
        lighting_suggestions=lighting_suggestions,
        layout_tips=layout_tips,
        wallpaper_suitability=wallpaper_suit,
    )


# ─── Rule functions ───────────────────────────────────────────────────────────

def _color_guidance(size: str, ceiling: str) -> str:
    rules = {
        ("small",  "low"):      "Use very light, cool tones on all surfaces. Avoid dark feature walls — they compress the space further. Keep colour transitions seamless.",
        ("small",  "standard"): "Light neutrals and soft pastels open the room. A single accent wall in a mid-tone adds character without overwhelming.",
        ("small",  "high"):     "You can afford a slightly bolder wall tone — the height compensates. A dark ceiling can feel dramatic and surprisingly cosy.",
        ("medium", "low"):      "Mid-tone walls work well. Avoid high-contrast colour blocking. A warm neutral with a soft accent strikes the right balance.",
        ("medium", "standard"): "Full creative latitude. Complementary colour schemes, feature walls, and two-tone treatments all work here.",
        ("medium", "high"):     "Consider dramatic paint treatments — colour drenching, ombré walls, or deep jewel tones paired with lighter trim.",
        ("large",  "low"):      "Bold, saturated wall colours anchor the space. Deep tones prevent the room feeling like a waiting room.",
        ("large",  "standard"): "Rich colours and layered tones. Zone different areas with distinct palettes tied by a common accent colour.",
        ("large",  "high"):     "No restrictions. Use the scale — grand statement colours, murals, gallery walls. Treat ceiling as a fifth design surface.",
    }
    return rules.get((size, ceiling), "Apply mid-tone base colours with complementary accents.")


def _furniture_scale(size: str) -> str:
    return {
        "small":  "Opt for slender-legged, multi-functional furniture. Avoid bulky sofas or over-sized case goods. Wall-mounted and floating pieces save floor space.",
        "medium": "Standard proportions work well. Mix a hero anchor piece (sofa, dining table) with lighter accent furniture. Leave breathing room between pieces.",
        "large":  "Use large-scale furniture to fill the room appropriately — oversized sofas, generous dining tables, statement armchairs. Avoid clustering small pieces.",
    }[size]


def _pattern_scale(size: str, shape: str) -> str:
    base = {
        "small":  "Small-scale, low-contrast patterns only. Micro geometrics, fine stripes, or plain textures. Avoid large florals or bold repeats.",
        "medium": "Medium-scale patterns (10–20 cm repeats) work well. Geometric, botanical, and abstract prints all suit the proportions.",
        "large":  "Large-scale, bold patterns are ideal. Oversized florals, wide stripes, grand geometric repeats — scale up to match the room.",
    }[size]

    if shape == "elongated":
        base += " Horizontal patterns or wide-repeat wallpapers help optically shorten the long axis."
    elif shape == "square":
        base += " Diagonal or all-over patterns work best in square rooms — they prevent the space feeling boxy."
    return base


def _lighting_suggestions(size: str, ceiling: str, room_type: str) -> list[str]:
    tips = []

    if size == "small":
        tips += [
            "Use recessed ceiling lights rather than pendants to preserve headroom.",
            "Wall sconces and picture lights add ambient glow without floor footprint.",
            "Mirrors opposite windows double perceived light.",
        ]
    elif size == "medium":
        tips += [
            "Layer lighting: ambient (ceiling), task (table/floor lamps), accent (spotlights).",
            "A central pendant or chandelier anchors the room visually.",
        ]
    else:
        tips += [
            "Zone lighting per activity area — reading corner, dining zone, display shelves.",
            "Statement chandeliers or clustered pendants fill vertical space.",
            "Floor uplighters create dramatic wall wash effects.",
        ]

    if ceiling == "low":
        tips.append("Avoid hanging pendants — use flush or semi-flush mounts. Uplighting tricks the eye into reading the space as taller.")
    elif ceiling == "high":
        tips.append("Long-drop pendants and chandeliers capitalise on the ceiling height. Consider cove lighting to illuminate the ceiling plane.")

    room_lighting = {
        "kitchen":    "Under-cabinet task lighting is essential. Avoid single overhead source — creates shadows on work surfaces.",
        "bedroom":    "Bedside reading lights (wall-mounted or pendants) free up nightstand space. Avoid harsh overhead lighting in bedrooms.",
        "bathroom":   "Side-lit vanity mirrors eliminate facial shadows. Add a waterproof recessed light over the shower.",
        "study_room": "Daylight-spectrum desk lamp (5000K) reduces eye strain. Bias lighting behind monitor helps in darker hours.",
        "dining_room":"A dimmable chandelier directly over the table is the anchor. Dimmer switches are non-negotiable for dining rooms.",
    }
    if room_type in room_lighting:
        tips.append(room_lighting[room_type])

    return tips


def _layout_tips(size: str, shape: str, room_type: str) -> list[str]:
    tips = []

    shape_tips = {
        "square": [
            "Arrange furniture at 45° angle to soften the boxy feel.",
            "A round dining table or circular rug suits square rooms perfectly.",
        ],
        "rectangular": [
            "Float furniture away from walls to define the living zone.",
            "A large rug anchors the seating arrangement.",
        ],
        "elongated": [
            "Divide into two functional zones — e.g. seating + dining, or reading nook + work area.",
            "Use a console table or open shelving unit to visually break the length.",
        ],
    }
    tips += shape_tips.get(shape, [])

    room_tips = {
        "kitchen":    ["Work triangle (sink, stove, fridge) should be unobstructed.", "Island placement only works if there is >1.2m clearance on all sides."],
        "bedroom":    ["Bed should face the door but not be directly in line with it.", "Leave >60cm on both sides of the bed for circulation."],
        "study_room": ["Position desk perpendicular to window to avoid glare on screen.", "Back-to-wall desk placement creates privacy when on calls."],
        "dining_room":["Table should have >90cm clearance on all sides for chair movement.", "Mirror on one wall makes the dining area feel more expansive."],
        "bathroom":   ["Keep toilet away from direct line of sight from door.", "Frameless shower glass maximises visual space."],
    }
    tips += room_tips.get(room_type, [])

    return tips


def _wallpaper_suitability(size: str, ceiling: str, shape: str) -> str:
    if size == "small" and ceiling == "low":
        return "Feature wall only (behind bed/sofa). Avoid all-over wallpaper — it shrinks the room. Choose small, airy patterns on a light ground."
    elif size == "small":
        return "One feature wall works well. Vertical stripe or small repeat botanical on a pale ground adds character without overwhelming."
    elif size == "medium":
        return "All-over wallpaper is suitable. Medium-scale patterns, textured plains, or a bold feature wall all work."
    else:
        return "All-over wallpaper excels here. Large repeats, mural-style papers, and textured grasscloths are all appropriate."


# ─── Utility: dimension → size ────────────────────────────────────────────────
def get_size_category(length: float, width: float) -> str:
    area = length * width
    if area < SMALL_MAX:
        return "small"
    elif area < MEDIUM_MAX:
        return "medium"
    return "large"


def get_ceiling_class(height: float) -> str:
    if height <= CEILING_LOW:
        return "low"
    elif height <= CEILING_STANDARD:
        return "standard"
    return "high"
