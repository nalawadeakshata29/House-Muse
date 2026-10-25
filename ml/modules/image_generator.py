"""
modules/image_generator.py
Primary  : Hugging Face ZeroGPU Space (SDXL + ControlNet + LoRA)
Fallback : Stability AI (text-only generation if STABILITY_API_KEY set)
"""

import os
import time
import base64
import requests
from urllib.parse import quote
from typing import Optional, Union, Dict, Any

# Make sure ml/.env (HF_SPACE_ID, HF_API_TOKEN, ...) is loaded even if the entry
# point forgot to call load_dotenv() before importing this module.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except Exception:
    pass

# ── Credentials ───────────────────────────────────────────────────────────────
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY", "").strip()
HF_SPACE_ID       = os.environ.get("HF_SPACE_ID", "").strip()
HF_API_TOKEN      = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")

if not STABILITY_API_KEY:
    print("[Generator] INFO: STABILITY_API_KEY not set — Stability AI fallback disabled.")
if not HF_SPACE_ID:
    print("[Generator] INFO: HF_SPACE_ID not set — HF Space generation disabled.")

# ── Room visual style map (imported by caption_engine) ────────────────────────
ROOM_VISUAL_MAP = {
    "living_room":  "a cosy living room with sofa, coffee table and ambient lighting",
    "bedroom":      "a serene bedroom with bed, soft furnishings and warm lighting",
    "kitchen":      "a modern kitchen with clean countertops and natural light",
    "dining_room":  "an elegant dining room with table, chairs and pendant lighting",
    "bathroom":     "a fresh bathroom with clean fixtures and soft lighting",
    "study_room":   "a focused study room with desk, bookshelves and task lighting",
    "hallway":      "a welcoming hallway with console table and wall art",
    "balcony":      "a relaxing balcony with outdoor seating and plants",
    "home_office":  "a productive home office with desk, monitor and clean shelving",
    "kids_room":    "a playful kids room with bright colours and fun furniture",
}


# ── Stability AI backend (FREE 25 credits/month) ──────────────────────────────
def _generate_via_stability(positive_prompt: str, negative_prompt: str) -> dict:
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    payload = {
        "text_prompts": [
            {"text": positive_prompt, "weight": 1.0},
            {"text": negative_prompt, "weight": -1.0},
        ],
        "width":        1024,
        "height":       1024,
        "steps":        30,
        "samples":      1,
        "cfg_scale":    7.5,
        "style_preset": "photographic",
    }

    print("[Stability] Requesting image ...")
    try:
        resp = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers=headers, json=payload, timeout=60,
        )
        if resp.status_code == 200:
            artifacts = resp.json().get("artifacts", [])
            if artifacts:
                print("[Stability] Image generated successfully.")
                return {"success": True, "image_url": None,
                        "image_b64": artifacts[0]["base64"], "error": None}
            return {"success": False, "image_url": None, "image_b64": None,
                    "error": "Stability AI returned no artifacts."}
        elif resp.status_code == 401:
            return {"success": False, "image_url": None, "image_b64": None,
                    "error": "Invalid STABILITY_API_KEY (401). Check backend/.env"}
        elif resp.status_code == 402:
            return {"success": False, "image_url": None, "image_b64": None,
                    "error": "Stability AI credits exhausted. Top up at platform.stability.ai"}
        else:
            return {"success": False, "image_url": None, "image_b64": None,
                    "error": f"Stability AI error {resp.status_code}: {resp.text[:200]}"}
    except requests.exceptions.Timeout:
        return {"success": False, "image_url": None, "image_b64": None,
                "error": "Stability AI request timed out."}
    except Exception as e:
        return {"success": False, "image_url": None, "image_b64": None,
                "error": f"Stability AI exception: {e}"}


# ── Public API ─────────────────────────────────────────────────────────────────
def generate_room_image(
    positive_prompt: str,
    negative_prompt: str = "",
    image_input=None,
    style: str = "contemporary_indian",
    room_type: str = "living_room",
) -> dict:
    """
    1. Hugging Face Space (SDXL + ControlNet + LoRA) when image_input is provided.
    2. Fall back to Stability AI only when no image_input is provided (text prompt only).
    """
    try:
        from modules.caption_engine import inject_room_furniture_into_prompt
        positive_prompt = inject_room_furniture_into_prompt(positive_prompt, room_type)
    except Exception:
        pass

    if image_input:
        if not HF_SPACE_ID:
            return {
                "success":     False,
                "image_url":   None,
                "image_b64":   None,
                "prompt":      positive_prompt,
                "error":       "HF_SPACE_ID is not set in ml/.env.",
                "status_code": 502,
            }

        try:
            from modules.hf_space_client import call_hf_space_redesign
            print(f"[Generator] Calling HF Space '{HF_SPACE_ID}' with input image...")
            hf_res = call_hf_space_redesign(
                space_id=HF_SPACE_ID,
                image_input=image_input,
                style=style,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                hf_token=HF_API_TOKEN,
            )
            if hf_res.get("success"):
                return {
                    "success":     True,
                    "image_url":   None,
                    "image_b64":   hf_res.get("image_b64"),
                    "image_path":  hf_res.get("image_path"),
                    "prompt":      positive_prompt,
                    "error":       None,
                    "status_code": 200,
                }
            return {
                "success":     False,
                "image_url":   None,
                "image_b64":   None,
                "prompt":      positive_prompt,
                "error":       hf_res.get("error", "Hugging Face Space generation failed"),
                "status_code": hf_res.get("status_code", 502),
            }
        except Exception as e:
            return {
                "success":     False,
                "image_url":   None,
                "image_b64":   None,
                "prompt":      positive_prompt,
                "error":       str(e),
                "status_code": 502,
            }

    # Text-only generation without uploaded room photo:
    if STABILITY_API_KEY:
        print("[Generator] Calling Stability AI for text-only prompt...")
        result = _generate_via_stability(positive_prompt, negative_prompt)
        result["prompt"] = positive_prompt
        result["status_code"] = 200 if result.get("success") else 502
        return result

    return {
        "success":     False,
        "image_url":   None,
        "image_b64":   None,
        "prompt":      positive_prompt,
        "error":       "Room image generation requires an uploaded room photo for Hugging Face Space redesign, or STABILITY_API_KEY for text generation.",
        "status_code": 400,
    }


def generate_multiple_variants(
    positive_prompt: str,
    negative_prompt: str = "",
    n_variants: int = 3,
    variant_notes=None,
    variant_labels=None,
    image_input=None,
    style: str = "contemporary_indian",
    room_type: str = "living_room",
) -> list:
    default_notes  = [
        "wide angle view showing full room",
        "close up corner detail showing furniture",
        "evening warm lighting cosy atmosphere",
    ]
    default_labels = ["Full Room View", "Detail View", "Evening Mood"]

    notes  = (variant_notes  or default_notes) [:n_variants]
    labels = (variant_labels or default_labels)[:n_variants]

    results = []
    for i in range(n_variants):
        note      = notes[i]  if i < len(notes)  else ""
        label     = labels[i] if i < len(labels) else f"Variant {i + 1}"
        extra_pos = f"{positive_prompt}, {note}" if note else positive_prompt
        result    = generate_room_image(extra_pos, negative_prompt, image_input=image_input, style=style, room_type=room_type)
        result["variant"]       = i + 1
        result["variant_label"] = label
        results.append(result)
    return results


# ── Refinement & Base Style Prompts ───────────────────────────────────────────
DEFAULT_STYLE_PROMPTS = {
    "coastal": (
        "coastal beach house interior design, completely refurnished with new coastal furniture, new seating, "
        "new lighting fixtures, sun-washed white walls, natural rattan chairs and sofa, "
        "sea blue and sandy beige accents, light linen curtains, breezy airy natural lighting, "
        "keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows, "
        "high end architectural digest photograph"
    ),
    "contemporary_indian": (
        "contemporary Indian interior design, completely refurnished with new contemporary Indian furniture, "
        "new designer seating, modern brass lighting fixtures, warm brass and copper accents, terracotta and warm ochre tones, "
        "modern Indian craftsmanship, handcrafted wood finishes, clean geometric lines, soft warm ambient lighting, "
        "keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows, "
        "high end architectural digest photograph"
    ),
    "eclectic": (
        "eclectic layered interior design, completely refurnished with new eclectic statement furniture, "
        "curated art wall, new seating, statement lighting fixtures, mix of vintage and contemporary furniture, "
        "rich textures, patterned accent rugs, bold cohesive color harmony, "
        "keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows, "
        "interior magazine photography"
    ),
    "industrial": (
        "industrial loft interior design, completely refurnished with new industrial furniture, new leather seating, "
        "exposed brick accent wall, black steel and iron metalwork, polished concrete floor, "
        "new reclaimed wood dining table, Edison bulb pendant lights, "
        "keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows, "
        "architectural photography"
    ),
    "mid_century": (
        "mid century modern interior design, completely refurnished with new mid-century modern furniture, "
        "new retro seating, iconic walnut credenza, tapered wooden legs, new lighting fixtures, "
        "mustard yellow and burnt orange accents, clean retro lines, large windows, warm diffused daylight, "
        "keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows"
    ),
    "scandavian": (
        "scandinavian interior design, completely refurnished with new scandinavian furniture, new minimalist seating, "
        "light blonde oak furniture, new modern lighting fixtures, soft neutral tones, minimal wabi-sabi elegance, "
        "cozy wool textiles, bright white walls, soft daylight, clutter free serene room, "
        "keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows"
    ),
    "shabby_chic_style": (
        "shabby chic interior design, completely refurnished with new shabby chic furniture, new vintage slipcovered seating, "
        "distressed painted vintage wood furniture, new chandeliers and lighting fixtures, soft pastel florals, antique lace, "
        "cozy romantic decor, soft warm lighting, charming lived-in French country aesthetic, "
        "keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows"
    ),
    "victorian": (
        "victorian interior design, completely refurnished with new victorian furniture, tufted jewel tone velvet seating, "
        "ornate crown moldings, dark mahogany woodwork, crystal chandelier and wall sconces, "
        "rich dramatic heritage atmosphere, "
        "keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows, "
        "professional interior photography"
    ),
    "vintage_indian": (
        "vintage Indian haveli interior, completely refurnished with new vintage Indian furniture, new carved seating, "
        "intricate carved jaali woodwork, antique brass artifacts, new heritage brass lantern lighting fixtures, "
        "rich indigo and terracotta hues, ornate teak furniture, traditional block-printed fabrics, heritage palace warmth, "
        "keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows"
    ),
}

_STYLE_MAP = {
    "scandinavian": "scandavian",
    "scandavian": "scandavian",
    "mid_century_modern": "mid_century",
    "mid-century-modern": "mid_century",
    "mid_century": "mid_century",
    "mid-century": "mid_century",
    "shabby_chic": "shabby_chic_style",
    "shabby-chic": "shabby_chic_style",
    "shabby_chic_style": "shabby_chic_style",
    "contemporary_indian": "contemporary_indian",
    "contemporary-indian": "contemporary_indian",
    "vintage_indian": "vintage_indian",
    "vintage-indian": "vintage_indian",
}


def get_base_style_prompt(style: str = "contemporary_indian", room_type: str = "living_room") -> str:
    """Returns the canonical base style prompt with furniture injection for room_type."""
    norm_style = style.lower().strip().replace(" ", "_") if style else "contemporary_indian"
    resolved_key = _STYLE_MAP.get(norm_style, norm_style)
    base_prompt = DEFAULT_STYLE_PROMPTS.get(
        resolved_key,
        f"high-end interior design photograph of a {norm_style} room, elegant, photorealistic, 8k"
    )
    try:
        from modules.caption_engine import inject_room_furniture_into_prompt
        base_prompt = inject_room_furniture_into_prompt(base_prompt, room_type)
    except Exception:
        pass
    return base_prompt


def _generate_img2img_via_stability(
    init_image,
    positive_prompt: str,
    negative_prompt: str = "",
    image_strength: float = 0.35,
) -> dict:
    if not STABILITY_API_KEY:
        return {"success": False, "image_url": None, "image_b64": None, "error": "No STABILITY_API_KEY set"}

    img_bytes = None
    if isinstance(init_image, bytes):
        img_bytes = init_image
    elif isinstance(init_image, str):
        if os.path.exists(init_image):
            with open(init_image, "rb") as f:
                img_bytes = f.read()
        elif init_image.startswith("http://") or init_image.startswith("https://"):
            try:
                r = requests.get(init_image, timeout=30)
                if r.status_code == 200:
                    img_bytes = r.content
            except Exception:
                pass
        else:
            try:
                b64_str = init_image.split(",", 1)[1] if "," in init_image else init_image
                img_bytes = base64.b64decode(b64_str)
            except Exception:
                pass
    elif hasattr(init_image, "read"):
        if hasattr(init_image, "seek"):
            init_image.seek(0)
        img_bytes = init_image.read()

    if not img_bytes:
        return {"success": False, "image_url": None, "image_b64": None, "error": "Could not read init_image bytes"}

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "application/json",
    }
    data = {
        "init_image_mode": "IMAGE_STRENGTH",
        "image_strength": float(image_strength),
        "text_prompts[0][text]": positive_prompt,
        "text_prompts[0][weight]": 1.0,
        "cfg_scale": 7.5,
        "samples": 1,
        "steps": 30,
    }
    if negative_prompt:
        data["text_prompts[1][text]"] = negative_prompt
        data["text_prompts[1][weight]"] = -1.0

    files = {"init_image": ("init.jpg", img_bytes, "image/jpeg")}

    try:
        resp = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image",
            headers=headers,
            data=data,
            files=files,
            timeout=60,
        )
        if resp.status_code == 200:
            artifacts = resp.json().get("artifacts", [])
            if artifacts:
                return {"success": True, "image_url": None, "image_b64": artifacts[0]["base64"], "error": None}
            return {"success": False, "image_url": None, "image_b64": None, "error": "Stability AI returned no artifacts"}
        else:
            return {"success": False, "image_url": None, "image_b64": None, "error": f"Stability AI error {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"success": False, "image_url": None, "image_b64": None, "error": str(e)}


def refine_room_image(
    positive_prompt: str,
    negative_prompt: str = "",
    init_image=None,
    original_image=None,
    instruction: str = "",
    style: str = "contemporary_indian",
    strength: Optional[float] = None,
    room_type: str = "living_room",
    image_id: Optional[str] = None,
) -> dict:
    """
    Refines a previous generated room image using Hugging Face Space img2img exclusively:
    1. Checks hash cache (image_id + instruction + settings) to avoid unnecessary GPU compute.
    2. Uses latest generated image as init image, and original room photo / depth map as ControlNet condition.
    3. Prompts with the instruction plus: 'keep the room's walls, windows, floor and ceiling unchanged'.
    4. Capped at REFINE_MAX_SIDE (768), REFINE_STEPS (25), and REFINE_STRENGTH (0.50).
    5. Fails immediately on any HF error with proper status code (429, 503, 502) — NO retries, NO fallbacks.
    """
    if not init_image:
        return {
            "success": False,
            "image_url": None,
            "image_b64": None,
            "prompt": positive_prompt,
            "instruction": instruction,
            "error": "No init image provided for refinement.",
            "status_code": 400,
        }

    if not HF_SPACE_ID:
        return {
            "success": False,
            "image_url": None,
            "image_b64": None,
            "prompt": positive_prompt,
            "instruction": instruction,
            "error": "HF_SPACE_ID is not set in ml/.env.",
            "status_code": 502,
        }

    # Resolve settings from environment
    refine_strength = float(os.environ.get("REFINE_STRENGTH", strength if strength is not None else 0.50))
    refine_steps = int(os.environ.get("REFINE_STEPS", 25))
    refine_max_side = int(os.environ.get("REFINE_MAX_SIDE", 768))

    settings_dict = {
        "strength": refine_strength,
        "steps": refine_steps,
        "max_side": refine_max_side,
        "style": style,
    }

    # Cache check by hash (image_id + instruction + settings)
    if image_id:
        try:
            from modules.image_cache import get_cached_refine_result
            cached_res = get_cached_refine_result(image_id, instruction, settings_dict)
            if cached_res:
                print(f"[Generator] Cache hit for refine (image_id={image_id}). Bypassing GPU call.")
                cached_res["status_code"] = 200
                return cached_res
        except Exception as cache_err:
            print(f"[Generator] Cache lookup warning: {cache_err}")

    # Build prompt ensuring user's instruction and structural preservation clause
    preserve_clause = "keep the room's walls, windows, floor and ceiling unchanged"
    instruction_clean = instruction.strip()

    if preserve_clause.lower() not in instruction_clean.lower():
        refined_instruction = f"{instruction_clean}, {preserve_clause}" if instruction_clean else preserve_clause
    else:
        refined_instruction = instruction_clean

    if refined_instruction.lower() not in positive_prompt.lower():
        full_prompt = f"{positive_prompt}, {refined_instruction}" if positive_prompt else refined_instruction
    else:
        full_prompt = positive_prompt

    try:
        from modules.hf_space_client import call_hf_space_refine
        print(f"[Generator] Calling HF Space '{HF_SPACE_ID}' for refinement (strength={refine_strength}, steps={refine_steps}, max_side={refine_max_side})...")
        hf_res = call_hf_space_refine(
            space_id=HF_SPACE_ID,
            image_input=init_image,
            style=style,
            instruction=refined_instruction,
            original_image_input=original_image,
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            strength=refine_strength,
            controlnet_scale=0.55,
            lora_scale=0.80,
            num_inference_steps=refine_steps,
            guidance_scale=7.5,
            hf_token=HF_API_TOKEN,
            max_side=refine_max_side,
        )

        if hf_res.get("success"):
            out_result = {
                "success": True,
                "image_url": None,
                "image_b64": hf_res.get("image_b64"),
                "image_path": hf_res.get("image_path"),
                "prompt": full_prompt,
                "instruction": instruction,
                "error": None,
                "status_code": 200,
            }
            # Save into refine hash cache
            if image_id:
                try:
                    from modules.image_cache import save_cached_refine_result
                    save_cached_refine_result(image_id, instruction, settings_dict, out_result)
                except Exception as save_err:
                    print(f"[Generator] Cache save warning: {save_err}")
            return out_result

        return {
            "success": False,
            "image_url": None,
            "image_b64": None,
            "prompt": full_prompt,
            "instruction": instruction,
            "error": hf_res.get("error", "Refinement failed."),
            "status_code": hf_res.get("status_code", 502),
        }

    except Exception as e:
        from modules.hf_space_client import classify_hf_error
        status_code, clean_msg = classify_hf_error(e)
        return {
            "success": False,
            "image_url": None,
            "image_b64": None,
            "prompt": full_prompt,
            "instruction": instruction,
            "error": clean_msg,
            "status_code": status_code,
        }