"""
modules/caption_engine.py
════════════════════════════════════════════════════════════════════════════════
BLIP Encoder-Decoder — Room Image → Semantic Caption → Diffusion Prompt

Pipeline:
    User Room Image
         ↓
    BLIP Encoder  (ViT — Vision Transformer)
         ↓         reads visual features from image
    BLIP Decoder  (Language Model)
         ↓         generates natural language description
    Caption       "a living room with a beige sofa and wooden coffee table"
         ↓
    Prompt Builder (combines caption + style + palette)
         ↓
    Stable Diffusion API → Final Styled Image

Why BLIP over raw prompt engineering:
    - Captures actual objects/colours FROM the photo automatically
    - Decoder output is grounded in real visual content
    - Much more accurate than guessing from class labels alone
    - Free via HuggingFace Inference API (no GPU, no cost)

Setup:
    1. Sign up free at https://huggingface.co
    2. Get token: https://huggingface.co/settings/tokens
    3. Add to backend/.env:
       HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

Model used: Salesforce/blip-image-captioning-large
    - Free on HuggingFace
    - Runs on their servers (no local GPU needed)
    - Response time: 3–8 seconds
════════════════════════════════════════════════════════════════════════════════
"""

import os
import base64
import requests
import time
from pathlib import Path


# ── HuggingFace Inference API config ─────────────────────────────────────────
BLIP_MODEL_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
HF_TOKEN       = os.getenv("HF_API_TOKEN", "")


# ── Style descriptors (normalized for all frontend style keys) ────────────────
STYLE_PROMPT_MAP = {
    "scandinavian":        "scandinavian interior design, wabi-sabi aesthetic, natural blonde oak wood textures, linen textiles, zen minimalism, muted warm tones, handcrafted details",
    "scandavian":          "scandinavian interior design, wabi-sabi aesthetic, natural blonde oak wood textures, linen textiles, zen minimalism, muted warm tones, handcrafted details",
    "japandi":             "japandi interior design, wabi-sabi aesthetic, natural wood textures, linen textiles, zen minimalism, muted warm tones, handcrafted details",
    "contemporary_indian": "contemporary Indian interior design, warm brass and copper accents, terracotta tones, handcrafted sheesham wood finishes, clean geometric lines, soft warm ambient lighting",
    "vintage_indian":      "vintage Indian haveli interior, intricate carved jaali woodwork, indigo and terracotta colours, antique brass artifacts, ornate carved teak furniture, heritage textiles",
    "coastal":             "coastal beach house interior, sun-washed white walls, natural rattan furniture, light linen curtains, sea blue and sandy beige accents, breezy natural daylight",
    "industrial":          "industrial loft interior design, exposed brick accent wall, black steel and iron metalwork, polished concrete floor, reclaimed wood table, Edison bulb pendant lights",
    "mid_century":         "mid century modern interior design, iconic walnut credenza, tapered wooden legs, mustard yellow and burnt orange accents, retro geometric lines, large windows, warm diffused daylight",
    "mid_century_modern":  "mid century modern interior design, iconic walnut credenza, tapered wooden legs, mustard yellow and burnt orange accents, retro geometric lines, large windows, warm diffused daylight",
    "eclectic":            "eclectic layered bohemian interior, curated art gallery wall, mix of vintage and contemporary statement furniture, rich textures, patterned accent rugs, bold cohesive color harmony",
    "bohemian":            "eclectic layered bohemian interior, curated art gallery wall, mix of vintage and contemporary statement furniture, rich textures, patterned accent rugs, bold cohesive color harmony",
    "shabby_chic_style":   "shabby chic cottagecore interior, vintage slipcovered seating, distressed painted wood furniture, crystal chandelier, soft pastel florals, antique lace, cozy romantic charm",
    "victorian":           "victorian heritage interior, tufted jewel tone velvet seating, ornate crown moldings, dark mahogany woodwork, crystal chandelier and wall sconces, rich dramatic atmosphere",
}

LIGHTING_MAP = {
    "bright": "flooded with natural daylight, bright and airy, sunlit",
    "dim":    "warm ambient lighting, soft lamp glow, intimate atmosphere",
}

SIZE_MAP = {
    "small":  "compact and cosy room, clever use of space",
    "medium": "well-proportioned room, balanced and comfortable",
    "large":  "spacious and grand room, generous proportions",
}

# ── Comprehensive Room Furniture Requirements (Explicitly adds seating, tables, lighting, rugs, wall art, plants) ─
ROOM_FURNITURE_REQUIREMENTS = {
    "living_room": (
        "fully furnished with a comfortable stylish sofa, accent armchairs, wooden coffee table, "
        "designer ceiling pendant lighting, warm floor lamp, plush textured area rug, curated framed wall art, lush potted indoor plants"
    ),
    "bedroom": (
        "fully furnished with a king bed, upholstered headboard, matching nightstands with bedside lamps, "
        "cozy duvet and throw pillows, wooden dresser, plush soft area rug, curated framed wall art, designer lighting, potted indoor plants"
    ),
    "kitchen": (
        "fully furnished with custom fitted cabinetry, marble stone countertop, breakfast bar with upholstered bar stools, "
        "designer pendant lights, woven runner rug, floating wall shelves, potted kitchen herbs"
    ),
    "dining_room": (
        "fully furnished with an elegant solid dining table, comfortable dining chairs, statement chandelier lighting, "
        "sideboard credenza, large patterned area rug, curated framed wall art, indoor plants"
    ),
    "bathroom": (
        "fully furnished with a luxury vanity cabinet with stone top, large mirror with modern sconce lighting, "
        "designer fixtures, plush bath rug, wall shelving, potted indoor plants"
    ),
    "study_room": (
        "fully furnished with a solid wood executive desk, ergonomic leather office chair, cozy reading armchair, "
        "tall bookshelves, task desk lamp, woven area rug, curated framed wall art, potted indoor plants"
    ),
}

DEFAULT_NEGATIVE_PROMPT = (
    "empty room, bare room, unfurnished, vacant room, empty floor, no furniture, sparse room, missing furniture, unfinished room, "
    "ugly, distorted, blurry, low resolution, bad proportions, unnatural geometry, distorted walls, "
    "crooked windows, cartoon, 3d render, illustration, painting, flat, people, person, human, "
    "watermark, text, signature, logo, oversaturated, noisy, artifacts, dark, gloomy, "
    "no bookshelf covering window, no wall replacing window, no blocked windows"
)

def get_room_furniture_requirements(room_type: str) -> str:
    """Return required furniture clause for room_type, normalizing common aliases."""
    if not room_type:
        return ROOM_FURNITURE_REQUIREMENTS["living_room"]
    key = room_type.lower().strip().replace(" ", "_").replace("-", "_")
    alias_map = {
        "living": "living_room",
        "bed": "bedroom",
        "bath": "bathroom",
        "dining": "dining_room",
        "study": "study_room",
        "office": "study_room",
        "home_office": "study_room",
    }
    key = alias_map.get(key, key)
    return ROOM_FURNITURE_REQUIREMENTS.get(key, ROOM_FURNITURE_REQUIREMENTS["living_room"])


def inject_room_furniture_into_prompt(prompt: str, room_type: str = "living_room", recommended_items: list = None) -> str:
    """
    Ensure prompt explicitly contains room furniture requirement and architectural preservation.
    """
    req = get_room_furniture_requirements(room_type)
    extra_items = ""
    if recommended_items:
        clean = [str(x).strip() for x in recommended_items if str(x).strip()]
        if clean:
            extra_items = f"featuring {', '.join(clean[:4])}, "

    arch_preservation = "keep the room's walls, windows, floor and ceiling exactly in place, maintain room structural geometry and openings, do not cover or replace windows"

    modified = prompt
    if req and req not in modified:
        modified = f"{modified}, {req}"
    if extra_items and extra_items not in modified:
        modified = f"{modified}, {extra_items}"
    if arch_preservation not in modified:
        modified = f"{modified}, {arch_preservation}"

    return modified




# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — BLIP ENCODER-DECODER: Image → Caption
# ══════════════════════════════════════════════════════════════════════════════

def generate_caption(image_bytes: bytes, max_retries: int = 3) -> dict:
    """
    Send room image to BLIP via HuggingFace Inference API.
    BLIP encoder reads image features, decoder generates a natural description.

    Args:
        image_bytes : raw image bytes (JPG/PNG)
        max_retries : retry if model is loading (cold start ~20s)

    Returns:
        {
            "success":  bool,
            "caption":  str,    # e.g. "a living room with grey sofa and wooden floor"
            "error":    str | None,
            "method":   str,    # "blip_api" | "fallback"
        }
    """
    if not HF_TOKEN:
        return {
            "success": False,
            "caption": "",
            "error":   "HF_API_TOKEN not set in .env file",
            "method":  "none",
        }

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    for attempt in range(max_retries):
        try:
            print(f"[BLIP] Generating caption (attempt {attempt + 1}/{max_retries})...")

            response = requests.post(
                BLIP_MODEL_URL,
                headers=headers,
                data=image_bytes,       # send raw image bytes directly
                timeout=45,
            )

            if response.status_code == 503:
                # Model is loading (cold start) — wait and retry
                wait_time = 20 if attempt == 0 else 10
                print(f"[BLIP] Model loading, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code == 200:
                result  = response.json()
                caption = ""

                # HF returns list of dicts: [{"generated_text": "..."}]
                if isinstance(result, list) and result:
                    caption = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    caption = result.get("generated_text", "")

                if caption:
                    print(f"[BLIP] Caption: '{caption}'")
                    return {
                        "success": True,
                        "caption": caption,
                        "error":   None,
                        "method":  "blip_api",
                    }

            print(f"[BLIP] HTTP {response.status_code}: {response.text[:200]}")

        except requests.Timeout:
            print(f"[BLIP] Timeout on attempt {attempt + 1}")
        except Exception as e:
            print(f"[BLIP] Error: {e}")

        if attempt < max_retries - 1:
            time.sleep(5)

    return {
        "success": False,
        "caption": "",
        "error":   "BLIP API failed after retries",
        "method":  "none",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — PROMPT BUILDER: Caption + Style + Palette → Diffusion Prompt
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt_from_caption(
    caption:           str,
    styles:            list[str],
    wall_color_name:   str,
    accent_color_name: str,
    size_category:     str,
    lighting:          str,
    dominant_colors:   list[dict] = None,
    room_type:         str = "living_room",
    recommended_items: list = None,
) -> tuple[str, str]:
    """
    Combine caption with style, explicit furniture, and preservation clauses.
    """
    clean_caption = (caption or "").lower()
    for bad_term in ["empty room", "unfurnished room", "bare room", "empty floor", "empty", "unfurnished", "bare"]:
        clean_caption = clean_caption.replace(bad_term, "room")
    clean_caption = " ".join(clean_caption.split()).strip().strip(",")

    style_key = (styles[0] if styles else "contemporary_indian").lower().strip()
    style_text = STYLE_PROMPT_MAP.get(style_key, STYLE_PROMPT_MAP.get(style_key.replace(" ", "_"), "contemporary interior design"))

    light_desc = LIGHTING_MAP.get(lighting, "natural lighting")
    size_desc  = SIZE_MAP.get(size_category, "medium sized room")

    colour_note = f"walls in {wall_color_name}, {accent_color_name} accents"
    if dominant_colors:
        dom = _hex_to_rough_name(dominant_colors[0]["hex"])
        colour_note += f", dominant {dom} colour palette"

    req_furniture = get_room_furniture_requirements(room_type)
    furniture_clause = f"{req_furniture}, " if req_furniture else ""

    reco_note = ""
    if recommended_items:
        clean = [str(x).strip() for x in recommended_items if str(x).strip()]
        if clean:
            reco_note = f"styled with curated designer pieces including {', '.join(clean[:4])}, "

    arch_clause = "keep the room's walls, windows, doors, floor and ceiling exactly in place, maintain structural architecture and openings, do not cover or replace windows"

    caption_segment = f"{clean_caption}, " if clean_caption and clean_caption != "room" else ""

    positive = (
        f"ultra realistic interior design photograph, {style_text}, "
        f"{caption_segment}"
        f"{furniture_clause}"
        f"{reco_note}"
        f"{colour_note}, {size_desc}, {light_desc}, "
        f"{arch_clause}, "
        f"professional architectural photography, architectural digest quality, "
        f"8k resolution, photorealistic, beautifully styled, soft natural shadows, depth of field, "
        f"no people, no text"
    )

    return positive, DEFAULT_NEGATIVE_PROMPT


def _hex_to_rough_name(hex_color: str) -> str:
    """Convert hex to a rough colour name for prompt use."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    if r > 200 and g > 180 and b > 150: return "warm cream"
    if r > 180 and g > 140 and b <  90: return "caramel"
    if r <  90 and g > 130 and b <  90: return "sage green"
    if r > 160 and g <  90 and b <  70: return "terracotta"
    if r <  70 and g <  70 and b > 140: return "navy blue"
    if r > 160 and g > 130 and b <  70: return "golden"
    if r <  80 and g <  60 and b <  60: return "dark charcoal"
    if r > 180 and g > 180 and b > 180: return "light grey"
    return "neutral"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — FALLBACK: Build prompt WITHOUT BLIP (if API unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt_fallback(
    room_type:        str,
    styles:           list[str],
    wall_color_name:  str,
    accent_color_name: str,
    size_category:    str,
    lighting:         str,
    detected_objects: list[dict] = None,
    dominant_colors:  list[dict] = None,
    recommended_items: list = None,
) -> tuple[str, str]:
    """
    Build prompt from structured data with explicit furniture, lighting, rugs, wall art, plants.
    """
    ROOM_VISUAL_MAP = {
        "living_room":  "living room",
        "bedroom":      "bedroom",
        "kitchen":      "kitchen",
        "dining_room":  "dining room",
        "bathroom":     "bathroom",
        "study_room":   "home office study room",
    }
    room_desc  = ROOM_VISUAL_MAP.get(room_type, "living room")
    style_key = (styles[0] if styles else "contemporary_indian").lower().strip()
    style_text = STYLE_PROMPT_MAP.get(style_key, STYLE_PROMPT_MAP.get(style_key.replace(" ", "_"), f"{style_key} interior design"))
    light_desc  = LIGHTING_MAP.get(lighting, "natural lighting")
    size_desc   = SIZE_MAP.get(size_category, "medium sized room")
    colour_note = f"walls in {wall_color_name}, {accent_color_name} accent pieces"

    # Add detected objects if available
    obj_note = ""
    if detected_objects:
        obj_names = [d.get("interior","").replace("_"," ")
                     for d in detected_objects[:4]
                     if d.get("interior")]
        if obj_names:
            obj_note = f"incorporating {', '.join(obj_names)}, "

    req_furniture = get_room_furniture_requirements(room_type)
    furniture_clause = f"{req_furniture}, " if req_furniture else ""

    reco_note = ""
    if recommended_items:
        clean = [str(x).strip() for x in recommended_items if str(x).strip()]
        if clean:
            reco_note = f"styled with curated pieces including {', '.join(clean[:4])}, "

    arch_clause = "keep the room's walls, windows, doors, floor and ceiling exactly in place, maintain structural architecture and openings, do not cover or replace windows"

    positive = (
        f"ultra realistic interior design photograph of a {room_desc}, {style_text}, "
        f"{obj_note}"
        f"{furniture_clause}"
        f"{reco_note}"
        f"{colour_note}, "
        f"{size_desc}, {light_desc}, "
        f"{arch_clause}, "
        f"professional architectural photography, architectural digest magazine quality, "
        f"8k resolution, photorealistic, beautifully styled, "
        f"soft natural shadows, depth of field, no people, no text"
    )

    return positive, DEFAULT_NEGATIVE_PROMPT


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION — called from app.py
# ══════════════════════════════════════════════════════════════════════════════

def get_generation_prompt(
    image_bytes:      bytes | None,
    room_type:        str,
    styles:           list[str],
    wall_color:       str,
    accent_color:     str,
    size_category:    str,
    lighting:         str,
    detected_objects: list[dict] = None,
    dominant_colors:  list[dict] = None,
    recommended_items: list = None,
) -> dict:
    """
    Main entry point. Tries BLIP first, falls back to structured prompt.
    """
    wall_name   = _hex_to_rough_name(wall_color)
    accent_name = _hex_to_rough_name(accent_color)

    caption = None
    method  = "fallback"

    # Try BLIP if image available and token set (fast timeout, fail gracefully)
    if image_bytes and HF_TOKEN:
        result = generate_caption(image_bytes, max_retries=1)
        if result.get("success") and result.get("caption"):
            caption = result["caption"]
            method  = "blip"
            pos, neg = build_prompt_from_caption(
                caption, styles, wall_name, accent_name,
                size_category, lighting, dominant_colors,
                room_type=room_type,
                recommended_items=recommended_items,
            )
            return {
                "positive_prompt": pos,
                "negative_prompt": neg,
                "caption":         caption,
                "method":          method,
            }

    # Fallback: structured prompt from analysis data
    pos, neg = build_prompt_fallback(
        room_type, styles, wall_name, accent_name,
        size_category, lighting, detected_objects, dominant_colors,
        recommended_items=recommended_items,
    )

    return {
        "positive_prompt": pos,
        "negative_prompt": neg,
        "caption":         caption,
        "method":          method,
    }

