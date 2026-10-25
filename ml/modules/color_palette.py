"""
color_palette.py
────────────────
Module 2 — Color Palette Generation
Inputs : dominant colors (from K-Means), lighting info, room size
Outputs: wall color, accent color, ceiling color, full palette with harmonies
"""

import colorsys
import numpy as np
from typing import Literal


# ─── Colour Harmony Types ─────────────────────────────────────────────────────
HarmonyType = Literal["complementary", "analogous", "triadic", "split_complementary"]


# ─── Helpers: RGB ↔ HSV ↔ Hex ────────────────────────────────────────────────
def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def lighten(r: int, g: int, b: int, amount: float = 0.25) -> str:
    h, s, v = rgb_to_hsv(r, g, b)
    v = min(1.0, v + amount)
    s = max(0.0, s - amount * 0.4)
    return rgb_to_hex(*hsv_to_rgb(h, s, v))


def darken(r: int, g: int, b: int, amount: float = 0.20) -> str:
    h, s, v = rgb_to_hsv(r, g, b)
    v = max(0.0, v - amount)
    return rgb_to_hex(*hsv_to_rgb(h, s, v))


def desaturate(r: int, g: int, b: int, amount: float = 0.35) -> str:
    h, s, v = rgb_to_hsv(r, g, b)
    s = max(0.0, s - amount)
    return rgb_to_hex(*hsv_to_rgb(h, s, v))


# ─── Harmony Generators ───────────────────────────────────────────────────────
def complementary(h: float, s: float, v: float) -> list[str]:
    """180° opposite hue."""
    comp_h = (h + 0.5) % 1.0
    return [
        rgb_to_hex(*hsv_to_rgb(comp_h, s, v)),
        rgb_to_hex(*hsv_to_rgb(comp_h, s * 0.7, min(v + 0.15, 1.0))),
    ]


def analogous(h: float, s: float, v: float) -> list[str]:
    """±30° neighbours."""
    return [
        rgb_to_hex(*hsv_to_rgb((h + 0.083) % 1.0, s, v)),
        rgb_to_hex(*hsv_to_rgb((h - 0.083) % 1.0, s, v)),
        rgb_to_hex(*hsv_to_rgb((h + 0.167) % 1.0, s * 0.8, min(v + 0.1, 1.0))),
    ]


def triadic(h: float, s: float, v: float) -> list[str]:
    """120° splits."""
    return [
        rgb_to_hex(*hsv_to_rgb((h + 0.333) % 1.0, s, v)),
        rgb_to_hex(*hsv_to_rgb((h + 0.667) % 1.0, s, v)),
    ]


def split_complementary(h: float, s: float, v: float) -> list[str]:
    """150° + 210° from base."""
    return [
        rgb_to_hex(*hsv_to_rgb((h + 0.417) % 1.0, s, v)),
        rgb_to_hex(*hsv_to_rgb((h + 0.583) % 1.0, s, v)),
    ]


# ─── Neutral Balancing ────────────────────────────────────────────────────────
def generate_neutrals(base_rgb: tuple[int, int, int]) -> dict:
    r, g, b = base_rgb
    return {
        "off_white":      lighten(r, g, b, 0.55),
        "light_neutral":  lighten(r, g, b, 0.35),
        "mid_neutral":    desaturate(r, g, b, 0.5),
        "dark_neutral":   darken(r, g, b, 0.30),
        "deep_anchor":    darken(r, g, b, 0.50),
    }


# ─── Room-Size Adjustments ────────────────────────────────────────────────────
def _apply_room_size_rules(
    palette: dict,
    room_size: str,            # "small" | "medium" | "large"
    ceiling_height: str,       # "low" | "standard" | "high"
    lighting: str,             # "bright" | "dim"
) -> dict:
    """
    Adjust saturation/value of wall and accent colours based on
    room-size heuristics so the final palette feels proportional.
    """
    wall_rgb = hex_to_rgb(palette["wall_color"])
    h, s, v  = rgb_to_hsv(*wall_rgb)

    if room_size == "small":
        # Lighter, less saturated → room feels larger
        s = max(0.0, s - 0.20)
        v = min(1.0, v + 0.15)
        palette["pattern_guidance"] = (
            "Prefer subtle, small-scale patterns or plain walls. "
            "Avoid heavy textures that visually shrink the space."
        )
    elif room_size == "large":
        # Bolder, richer tones fill the space
        s = min(1.0, s + 0.10)
        v = max(0.0, v - 0.05)
        palette["pattern_guidance"] = (
            "Bold colours, deep textures, and large-scale patterns "
            "work well — they add warmth and prevent the room feeling cavernous."
        )
    else:
        palette["pattern_guidance"] = (
            "Medium-scale patterns and mid-tone colours balance the space effectively."
        )

    if ceiling_height == "low":
        palette["ceiling_color"]   = lighten(*hex_to_rgb(palette["wall_color"]), 0.45)
        palette["ceiling_guidance"] = (
            "Use a ceiling colour 2–3 shades lighter than the wall to "
            "draw the eye upward and create an illusion of height. "
            "Consider vertical stripes or tall furniture."
        )
    elif ceiling_height == "high":
        palette["ceiling_color"]   = desaturate(*hex_to_rgb(palette["wall_color"]), 0.3)
        palette["ceiling_guidance"] = (
            "A slightly deeper ceiling tone brings the room into balance "
            "and adds architectural drama."
        )
    else:
        palette["ceiling_color"]   = lighten(*hex_to_rgb(palette["wall_color"]), 0.30)
        palette["ceiling_guidance"] = "A classic off-white ceiling complements most wall colours."

    if lighting == "dim":
        # Push wall towards lighter/warmer
        v = min(1.0, v + 0.10)
        palette["lighting_note"] = (
            "Room has limited natural light — lighter wall tones and warm "
            "accent lighting will prevent the space feeling gloomy."
        )
    else:
        palette["lighting_note"] = (
            "Good natural light — you can afford richer, deeper wall tones "
            "without the room feeling closed in."
        )

    palette["wall_color"] = rgb_to_hex(*hsv_to_rgb(h, s, v))
    return palette


# ─── Master Palette Builder ───────────────────────────────────────────────────
def generate_palette(
    dominant_colors: list[dict],
    room_size:       str = "medium",    # small | medium | large
    ceiling_height:  str = "standard",  # low | standard | high
    lighting:        str = "bright",    # bright | dim
    style:           str = "Japandi",   # user's aesthetic choice
    harmony:         HarmonyType = "analogous",
) -> dict:
    """
    Full colour palette generation pipeline.

    Args:
        dominant_colors : list from extract_dominant_colors()
        room_size       : derived from L×W dimensions
        ceiling_height  : derived from height dimension
        lighting        : from detect_lighting()
        style           : user aesthetic preference
        harmony         : colour harmony type to apply

    Returns full palette dict with wall, accent, ceiling, harmonies, neutrals.
    """
    if not dominant_colors:
        raise ValueError("dominant_colors list is empty.")

    # Primary base = most dominant extracted colour
    base       = dominant_colors[0]
    base_rgb   = tuple(base["rgb"])
    h, s, v    = rgb_to_hsv(*base_rgb)

    # Secondary = 2nd most dominant (for accent)
    sec_rgb    = tuple(dominant_colors[1]["rgb"]) if len(dominant_colors) > 1 else base_rgb
    sec_h, sec_s, sec_v = rgb_to_hsv(*sec_rgb)

    # ── Harmony colours ──
    harmony_fn = {
        "complementary":       complementary,
        "analogous":           analogous,
        "triadic":             triadic,
        "split_complementary": split_complementary,
    }.get(harmony, analogous)

    harmony_colors = harmony_fn(h, s, v)

    # ── Wall & Accent ──
    wall_color   = desaturate(*base_rgb, 0.15)          # slightly muted for walls
    accent_color = harmony_colors[0] if harmony_colors else rgb_to_hex(*sec_rgb)

    # ── Neutrals ──
    neutrals = generate_neutrals(base_rgb)

    palette = {
        "wall_color":    wall_color,
        "accent_color":  accent_color,
        "ceiling_color": lighten(*base_rgb, 0.30),       # default; overridden below
        "harmony_type":  harmony,
        "harmony_colors": harmony_colors,
        "neutrals":       neutrals,
        "dominant_base":  base["hex"],
        "all_dominant":   [c["hex"] for c in dominant_colors],
        "style":          style,
        "ceiling_guidance": "",
        "pattern_guidance": "",
        "lighting_note":    "",
    }

    # ── Apply room-size + height + lighting rules ──
    palette = _apply_room_size_rules(palette, room_size, ceiling_height, lighting)

    # ── Style-specific tweaks ──
    palette = _apply_style_overlay(palette, style, base_rgb)

    return palette


# ─── Style Overlays ───────────────────────────────────────────────────────────
_STYLE_ACCENT_HINTS = {
    "Japandi":             {"s_offset": -0.15, "v_offset":  0.05, "warm": False},
    "Bohemian":            {"s_offset":  0.10, "v_offset": -0.05, "warm": True},
    "Mid-Century":         {"s_offset":  0.05, "v_offset":  0.00, "warm": True},
    "Contemporary Indian": {"s_offset":  0.15, "v_offset": -0.08, "warm": True},
    "Vintage Indian":      {"s_offset":  0.20, "v_offset": -0.12, "warm": True},
    "Art Deco":            {"s_offset": -0.05, "v_offset": -0.10, "warm": False},
    "Coastal":             {"s_offset": -0.20, "v_offset":  0.15, "warm": False},
    "Dark Academia":       {"s_offset":  0.00, "v_offset": -0.25, "warm": True},
    "Cottagecore":         {"s_offset": -0.10, "v_offset":  0.10, "warm": True},
    "Industrial":          {"s_offset": -0.25, "v_offset": -0.15, "warm": False},
}

def _apply_style_overlay(palette: dict, style: str, base_rgb: tuple) -> dict:
    hint = _STYLE_ACCENT_HINTS.get(style, {})
    if not hint:
        return palette

    h, s, v = rgb_to_hsv(*base_rgb)
    s = max(0.0, min(1.0, s + hint.get("s_offset", 0)))
    v = max(0.0, min(1.0, v + hint.get("v_offset", 0)))

    if hint.get("warm"):
        h = (h + 0.02) % 1.0       # nudge slightly warm

    palette["style_accent"] = rgb_to_hex(*hsv_to_rgb(h, s, v))
    palette["style_note"]   = _style_description(style)
    return palette


def _style_description(style: str) -> str:
    notes = {
        "Japandi":             "Wabi-sabi meets Scandinavian calm — muted tones, natural textures, purposeful negative space.",
        "Bohemian":            "Rich jewel tones, layered textiles, global patterns — maximalist warmth.",
        "Mid-Century":         "Warm woods, mustard yellows, burnt oranges — retro optimism with clean lines.",
        "Contemporary Indian": "Modern sensibility fused with Indian craft — block prints, brass accents, terracotta.",
        "Vintage Indian":      "Heritage havelis revived — deep magentas, indigo, intricate jaali patterns, antique brass.",
        "Art Deco":            "Geometric opulence — gold, black, emerald. Symmetry and glamour.",
        "Coastal":             "Bleached whites, sea blues, rattan — breezy and light.",
        "Dark Academia":       "Ink, mahogany, candlelight — moody intellectualism meets library aesthetics.",
        "Cottagecore":         "Floral ditsy prints, sage greens, buttery creams — pastoral romanticism.",
        "Industrial":          "Raw concrete, exposed steel, charcoal — urban loft energy.",
    }
    return notes.get(style, "A curated aesthetic blend tailored to your space.")
