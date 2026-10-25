"""
modules/hf_space_client.py — Programmatic Client for Hugging Face ZeroGPU Space

Usage in Flask / image_generator.py:
    from modules.hf_space_client import call_hf_space_redesign

    result = call_hf_space_redesign(
        space_id="your-username/house-muse-interior-sdxl",
        image_input="path/to/room.jpg", # or PIL.Image, or bytes
        style="contemporary_indian",
        hf_token=os.getenv("HF_API_TOKEN"),
    )
    # result["image_b64"], result["image_path"], result["success"]
"""

import os
import io
import base64
import tempfile
from typing import Optional, Dict, Any, Union, Tuple
from PIL import Image

try:
    from gradio_client import Client, handle_file
except ImportError:
    Client = None
    handle_file = None


# The Space's style dropdown only accepts these exact 9 values:
# 'contemporary_indian', 'vintage_indian', 'coastal', 'mid_century',
# 'scandavian', 'industrial', 'eclectic', 'shabby_chic_style', 'victorian'
VALID_HF_STYLES = {
    "contemporary_indian",
    "vintage_indian",
    "coastal",
    "mid_century",
    "scandavian",
    "industrial",
    "eclectic",
    "shabby_chic_style",
    "victorian",
}

_STYLE_ALIASES = {
    "scandinavian": "scandavian",
    "scandavian": "scandavian",
    "japandi": "scandavian",
    "scandinavian_japandi": "scandavian",
    "scandinavian-japandi": "scandavian",
    "wabi_sabi": "scandavian",
    "wabi-sabi": "scandavian",
    "shabby_chic": "shabby_chic_style",
    "shabby-chic": "shabby_chic_style",
    "shabby_chic_style": "shabby_chic_style",
    "cottagecore": "shabby_chic_style",
    "mid_century_modern": "mid_century",
    "mid-century": "mid_century",
    "mid-century-modern": "mid_century",
    "mid_century": "mid_century",
    "contemporary_indian": "contemporary_indian",
    "contemporary-indian": "contemporary_indian",
    "vintage_indian": "vintage_indian",
    "vintage-indian": "vintage_indian",
    "bohemian": "eclectic",
    "eclectic_bohemian": "eclectic",
    "eclectic": "eclectic",
    "coastal": "coastal",
    "industrial": "industrial",
    "victorian": "victorian",
    "art_deco": "contemporary_indian",
    "art-deco": "contemporary_indian",
    "dark_academia": "victorian",
    "dark-academia": "victorian",
}

def resolve_hf_style(style_input: str) -> str:
    raw = (style_input or "contemporary_indian").lower().strip()
    norm = raw.replace(" ", "_")
    if norm in _STYLE_ALIASES:
        return _STYLE_ALIASES[norm]
    if raw in _STYLE_ALIASES:
        return _STYLE_ALIASES[raw]
    if norm in VALID_HF_STYLES:
        return norm
    return "contemporary_indian"


def classify_hf_error(err: Union[Exception, str]) -> Tuple[int, str]:
    """
    Maps Hugging Face ZeroGPU / Space errors to HTTP status code and user-facing message.
    429: Quota exhausted / Rate limit
    503: Space sleeping / building / unavailable
    502: Other upstream error
    """
    err_str = str(err)
    err_lower = err_str.lower()

    if any(q in err_lower for q in ["quota", "429", "rate limit", "exceeded", "daily compute limit", "zerogpu quota"]):
        return 429, "Hugging Face quota exhausted, try again later"
    if any(s in err_lower for s in ["asleep", "sleeping", "paused", "503", "is loading", "building", "queue is full", "queue full", "unavailable"]):
        return 503, "Hugging Face Space is currently sleeping or starting up, try again shortly"
    return 502, f"Hugging Face Space error: {err_str}"


def call_hf_space_redesign(
    space_id: str,
    image_input: Union[str, bytes, Image.Image],
    style: str = "contemporary_indian",
    prompt: str = "",
    negative_prompt: str = "",
    controlnet_scale: float = 0.55,
    lora_scale: float = 0.80,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = -1,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls the Hugging Face Space via gradio_client.
    """
    if Client is None:
        return {
            "success": False,
            "image_path": None,
            "image_b64": None,
            "style": style,
            "status": None,
            "error": "gradio_client is not installed. Install it with: pip install gradio_client",
            "status_code": 502,
        }

    style = resolve_hf_style(style)
    token = hf_token or os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")

    temp_file_path = None
    try:
        if isinstance(image_input, str) and os.path.exists(image_input):
            target_path = image_input
        else:
            if isinstance(image_input, str):
                if "," in image_input:
                    image_input = image_input.split(",", 1)[1]
                img_bytes = base64.b64decode(image_input)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif isinstance(image_input, bytes):
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                pil_img.save(tmp.name, format="JPEG", quality=92)
                target_path = tmp.name
                temp_file_path = tmp.name

        try:
            client = Client(space_id, token=token, httpx_kwargs={"timeout": 300.0})
        except TypeError:
            try:
                client = Client(space_id, token=token)
            except TypeError:
                client = Client(space_id, hf_token=token)

        result = client.predict(
            image=handle_file(target_path),
            style=style,
            prompt=prompt,
            negative_prompt=negative_prompt,
            controlnet_scale=float(controlnet_scale),
            lora_scale=float(lora_scale),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed),
            api_name="/redesign",
        )

        output_image_path, status_text = result

        with open(output_image_path, "rb") as f:
            out_bytes = f.read()
        out_b64 = base64.b64encode(out_bytes).decode("utf-8")

        return {
            "success": True,
            "image_path": output_image_path,
            "image_b64": out_b64,
            "style": style,
            "status": status_text,
            "error": None,
            "status_code": 200,
        }

    except Exception as e:
        status_code, clean_msg = classify_hf_error(e)
        return {
            "success": False,
            "image_path": None,
            "image_b64": None,
            "style": style,
            "status": None,
            "error": clean_msg,
            "status_code": status_code,
            "raw_error": str(e),
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


def _prepare_temp_file(
    image_input: Union[str, bytes, Image.Image],
    max_side: Optional[int] = None,
) -> Tuple[str, Optional[str]]:
    """Converts image input to a local file path, capping long side to max_side if specified."""
    pil_img = None
    if isinstance(image_input, str) and os.path.exists(image_input):
        if max_side:
            try:
                pil_img = Image.open(image_input).convert("RGB")
            except Exception:
                return image_input, None
        else:
            return image_input, None

    if pil_img is None:
        if isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                import requests
                r = requests.get(image_input, timeout=30)
                r.raise_for_status()
                pil_img = Image.open(io.BytesIO(r.content)).convert("RGB")
            else:
                if "," in image_input:
                    image_input = image_input.split(",", 1)[1]
                img_bytes = base64.b64decode(image_input)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if max_side and max(pil_img.size) > max_side:
        w, h = pil_img.size
        scale = max_side / float(max(w, h))
        new_w, new_h = max(64, int(w * scale)), max(64, int(h * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_img.save(tmp.name, format="JPEG", quality=90)
        return tmp.name, tmp.name


def call_hf_space_refine(
    space_id: str,
    image_input: Union[str, bytes, Image.Image],
    style: str = "contemporary_indian",
    instruction: str = "",
    original_image_input: Optional[Union[str, bytes, Image.Image]] = None,
    prompt: str = "",
    negative_prompt: str = "",
    strength: float = 0.50,
    controlnet_scale: float = 0.55,
    lora_scale: float = 0.80,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = -1,
    hf_token: Optional[str] = None,
    max_side: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calls the Hugging Face Space for img2img refinement via gradio_client.
    Fails immediately on error (NO retries) and returns mapped HTTP status codes.
    """
    if Client is None:
        return {
            "success": False,
            "image_path": None,
            "image_b64": None,
            "style": style,
            "instruction": instruction,
            "status": None,
            "error": "gradio_client is not installed. Install it with: pip install gradio_client",
            "status_code": 502,
        }

    style = resolve_hf_style(style)
    token = hf_token or os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
    effective_max_side = max_side or int(os.environ.get("REFINE_MAX_SIDE", 768))

    temp_files = []
    try:
        target_path, temp_path = _prepare_temp_file(image_input, max_side=effective_max_side)
        if temp_path:
            temp_files.append(temp_path)

        target_orig_path = None
        if original_image_input:
            try:
                target_orig_path, temp_orig = _prepare_temp_file(original_image_input, max_side=effective_max_side)
                if temp_orig:
                    temp_files.append(temp_orig)
            except Exception:
                target_orig_path = None

        try:
            client = Client(space_id, token=token, httpx_kwargs={"timeout": 300.0})
        except TypeError:
            try:
                client = Client(space_id, token=token)
            except TypeError:
                client = Client(space_id, hf_token=token)

        # Call /refine endpoint. Strictly NO retries if an error occurs to save quota.
        result = client.predict(
            image=handle_file(target_path),
            style=style,
            instruction=instruction,
            original_image=handle_file(target_orig_path) if target_orig_path else None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=float(strength),
            controlnet_scale=float(controlnet_scale),
            lora_scale=float(lora_scale),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed),
            api_name="/refine",
        )

        output_image_path, status_text = result

        with open(output_image_path, "rb") as f:
            out_bytes = f.read()
        out_b64 = base64.b64encode(out_bytes).decode("utf-8")

        return {
            "success": True,
            "image_path": output_image_path,
            "image_b64": out_b64,
            "style": style,
            "instruction": instruction,
            "status": status_text,
            "error": None,
            "status_code": 200,
        }

    except Exception as e:
        status_code, clean_msg = classify_hf_error(e)
        return {
            "success": False,
            "image_path": None,
            "image_b64": None,
            "style": style,
            "instruction": instruction,
            "status": None,
            "error": clean_msg,
            "status_code": status_code,
            "raw_error": str(e),
        }
    finally:
        for p in temp_files:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


