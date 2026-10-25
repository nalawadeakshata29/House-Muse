"""
modules/controlnet_generator.py — Standalone SDXL + Depth ControlNet + LoRA Room Regeneration

Pure PyTorch + Diffusers implementation (hosting-agnostic).
Can run on any environment with an NVIDIA GPU (local, RunPod, EC2, Colab, etc.).

Features:
  - Base SDXL 1.0 + Depth ControlNet + FP16 VAE fix
  - Depth-Anything preprocessor for geometry & spatial preservation
  - Pre-loads all 9 style LoRA adapters into memory; switches instantly per request via set_adapters()
  - Zero serverless/platform-specific vendor locking
"""

import os
import io
import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_LORA_DIR = BASE_DIR / "models" / "lora"
if not LOCAL_LORA_DIR.exists():
    LOCAL_LORA_DIR = BASE_DIR / "models" / "loras"

# ── 9 LoRA files & mappings ───────────────────────────────────────────────────
LORA_FILES: Dict[str, str] = {
    "coastal": "coastal.safetensors",
    "contemporary_indian": "contemporary_indian.safetensors",
    "eclectic": "eclectic.safetensors",
    "industrial": "industrial.safetensors",
    "mid_century": "mid_century.safetensors",
    "scandavian": "scandavian.safetensors",
    "shabby_chic_style": "shabby_chic_style.safetensors",
    "victorian": "victorian.safetensors",
    "vintage_indian": "vintage_indian.safetensors",
}

STYLE_TO_ADAPTER: Dict[str, str] = {
    "coastal": "coastal",
    "contemporary_indian": "contemporary_indian",
    "contemporary-indian": "contemporary_indian",
    "eclectic": "eclectic",
    "industrial": "industrial",
    "mid_century": "mid_century",
    "mid-century": "mid_century",
    "mid_century_modern": "mid_century",
    "mid-century-modern": "mid_century",
    "scandinavian": "scandavian",
    "scandavian": "scandavian",
    "shabby_chic": "shabby_chic_style",
    "shabby-chic": "shabby_chic_style",
    "shabby_chic_style": "shabby_chic_style",
    "victorian": "victorian",
    "vintage_indian": "vintage_indian",
    "vintage-indian": "vintage_indian",
}

DEFAULT_STYLE_PROMPTS: Dict[str, str] = {
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
        "keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows"
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

DEFAULT_NEGATIVE_PROMPT = (
    "ugly, distorted, blurry, low resolution, bad proportions, unnatural geometry, distorted walls, "
    "crooked windows, cartoon, 3d render, illustration, painting, flat, people, person, human, "
    "watermark, text, signature, logo, oversaturated, noisy, artifacts, dark, gloomy, "
    "no bookshelf covering window, no wall replacing window, no blocked windows"
)


# ── Preprocessing & Geometry Extraction ────────────────────────────────────────

def prepare_image_for_sdxl(raw_image: Image.Image, target_max: int = 1024) -> Image.Image:
    """Resize input photo preserving aspect ratio and snapping dimensions to multiples of 64."""
    w, h = raw_image.size
    aspect = w / h

    if aspect >= 1:
        new_w = target_max
        new_h = int(target_max / aspect)
    else:
        new_h = target_max
        new_w = int(target_max * aspect)

    new_w = max(512, (new_w // 64) * 64)
    new_h = max(512, (new_h // 64) * 64)

    return raw_image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def extract_depth_map(depth_estimator, input_image: Image.Image) -> Image.Image:
    """Generate high-contrast depth map from the input room photo using Depth-Anything."""
    depth_output = depth_estimator(input_image)["depth"]
    depth_rgb = depth_output.convert("RGB")
    return depth_rgb.resize(input_image.size, Image.Resampling.LANCZOS)


def decode_base64_image(image_input: Any) -> Image.Image:
    """Convert base64 string, bytes, or PIL Image into an RGB PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, str):
        if "," in image_input:
            image_input = image_input.split(",", 1)[1]
        img_bytes = base64.b64decode(image_input)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def encode_image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 90) -> str:
    """Encode PIL Image to JPEG base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ── Model Pipeline Manager ─────────────────────────────────────────────────────

class RoomRedesignEngine:
    """
    Manages loading and running SDXL + ControlNet + 9 style LoRAs.
    Loads models into VRAM once, then switches LoRA adapters per generation.
    """

    def __init__(self, device: str = "cuda", lora_dir: Optional[Path] = None):
        self.device = device
        self.lora_dir = lora_dir or LOCAL_LORA_DIR
        self.loaded_adapters = set()
        self.pipe = None
        self.depth_estimator = None

    def load(self):
        """Load Depth-Anything, ControlNet Depth, SDXL base, and all 9 LoRA adapters."""
        import torch
        from transformers import pipeline
        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            ControlNetModel,
            AutoencoderKL,
            DPMSolverMultistepScheduler,
        )

        print(f"[RoomRedesignEngine] Initializing on {self.device}...")
        t0 = time.time()

        # 1. Depth-Anything estimator
        print("[Init] Loading Depth-Anything preprocessor...")
        self.depth_estimator = pipeline(
            "depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=self.device,
        )

        # 2. ControlNet Depth
        print("[Init] Loading SDXL Depth ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16,
        )

        # 3. SDXL VAE (FP16 fix)
        print("[Init] Loading SDXL FP16 VAE...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )

        # 4. SDXL Base Pipeline
        print("[Init] Loading SDXL Base Pipeline...")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(self.device)

        # DPM-Solver++ scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        self.pipe.vae.enable_tiling()

        # 5. Load all 9 LoRA adapters into memory
        print(f"[Init] Loading LoRA adapters from {self.lora_dir}...")
        for adapter_name, filename in LORA_FILES.items():
            lora_path = Path(self.lora_dir) / filename
            if lora_path.exists():
                try:
                    self.pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                    self.loaded_adapters.add(adapter_name)
                    print(f"  ✓ Loaded LoRA: '{adapter_name}' ({filename})")
                except Exception as e:
                    print(f"  ✗ Failed to load LoRA '{adapter_name}': {e}")
            else:
                print(f"  ✗ LoRA file not found: {lora_path}")

        print(f"[RoomRedesignEngine] Ready in {round(time.time() - t0, 2)}s.")
        return self

    def redesign(
        self,
        image: Any,
        style: str,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        controlnet_scale: float = 0.55,
        lora_scale: float = 0.80,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute room transformation using Depth ControlNet + LoRA + SDXL.

        Returns:
            {
                "success": bool,
                "image": PIL.Image,
                "image_b64": str,
                "style": str,
                "adapter_applied": str | None,
                "prompt": str,
                "seed": int,
                "duration_seconds": float,
            }
        """
        import torch

        t0 = time.time()

        # 1. Decode & preprocess image
        source_image = decode_base64_image(image)
        processed_image = prepare_image_for_sdxl(source_image)
        depth_map = extract_depth_map(self.depth_estimator, processed_image)

        # 2. Select LoRA adapter
        normalized_style = style.lower().strip()
        adapter_key = STYLE_TO_ADAPTER.get(normalized_style)

        if adapter_key and adapter_key in self.loaded_adapters:
            self.pipe.set_adapters([adapter_key], adapter_weights=[lora_scale])
            active_adapter = adapter_key
            print(f"[Inference] Set active LoRA '{adapter_key}' (weight={lora_scale})")
        else:
            self.pipe.disable_lora()
            active_adapter = None
            print(f"[Inference] No LoRA matched for '{style}'. Running base SDXL.")

        # 3. Prompt formulation
        if prompt and prompt.strip():
            final_prompt = prompt.strip()
        else:
            final_prompt = DEFAULT_STYLE_PROMPTS.get(
                adapter_key or normalized_style,
                f"high-end interior design photograph of a {normalized_style} redesigned room, completely refurnished with new {normalized_style} furniture, new seating, new lighting fixtures, keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows, elegant, photorealistic, 8k"
            )

        final_negative = negative_prompt or DEFAULT_NEGATIVE_PROMPT

        # 4. Seed
        seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 5. Run diffusion
        print(f"[Inference] Generating image ({num_inference_steps} steps, controlnet={controlnet_scale})...")
        result_image = self.pipe(
            prompt=final_prompt,
            negative_prompt=final_negative,
            image=depth_map,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        duration = round(time.time() - t0, 2)
        output_b64 = encode_image_to_base64(result_image)

        return {
            "success": True,
            "image": result_image,
            "image_b64": output_b64,
            "style": style,
            "adapter_applied": active_adapter,
            "prompt": final_prompt,
            "seed": seed,
            "duration_seconds": duration,
        }
