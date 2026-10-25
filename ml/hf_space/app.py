"""
app.py — Hugging Face Space (Gradio + ZeroGPU)
House Muse: SDXL + Depth ControlNet + 9 Style LoRAs

CRITICAL ZeroGPU RULE:
  'import spaces' MUST be the very first import before importing torch or diffusers.
"""

import spaces  # <-- Must be imported before torch!
import os
import io
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
from PIL import Image

import torch
import gradio as gr

# --- Workaround: gradio_client crashes on boolean JSON-schema nodes (api_info TypeError) ---
import gradio_client.utils as _gc_utils
_orig_schema_to_type = _gc_utils._json_schema_to_python_type
def _safe_schema_to_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_schema_to_type(schema, defs)
_gc_utils._json_schema_to_python_type = _safe_schema_to_type
_orig_get_type = _gc_utils.get_type
def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)
_gc_utils.get_type = _safe_get_type
print("[Patch] gradio_client bool-schema patch ACTIVE")
# ------------------------------------------------------------------------------------------
from transformers import pipeline
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. PATHS & LORA MAPPINGS
# ══════════════════════════════════════════════════════════════════════════════

# Locate local LoRA directory inside the Space repo
POSSIBLE_LORA_DIRS = [
    Path("models/loras"),
    Path("models/lora"),
    Path("loras"),
    Path(__file__).parent / "models" / "lora",
    Path(__file__).parent / "models" / "loras",
]

LORA_DIR = None
for candidate in POSSIBLE_LORA_DIRS:
    if candidate.exists():
        LORA_DIR = candidate
        break

if LORA_DIR is None:
    LORA_DIR = Path("models/loras")

# 9 trained style LoRAs
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

STYLE_CHOICES = [
    ("Contemporary Indian (Craft, clean lines, warm brass)", "contemporary_indian"),
    ("Vintage Indian (Haveli woodwork, terracotta, royal heritage)", "vintage_indian"),
    ("Coastal (Sun-washed white, natural rattan, airy daylight)", "coastal"),
    ("Mid-century Modern (Walnut credenza, tapered legs, retro geometry)", "mid_century"),
    ("Scandinavian (Blonde oak, serene neutrality, cozy wool)", "scandavian"),
    ("Industrial (Exposed brick, iron framing, concrete floors)", "industrial"),
    ("Eclectic (Curated art wall, layered textures, bold personality)", "eclectic"),
    ("Shabby Chic (Distressed vintage wood, soft romantic florals)", "shabby_chic_style"),
    ("Victorian (Ornate molding, jewel tones, crystal chandelier)", "victorian"),
]

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


# ══════════════════════════════════════════════════════════════════════════════
# 2. GLOBAL MODEL INITIALIZATION (Runs once when Space container starts)
# ══════════════════════════════════════════════════════════════════════════════

print("\n🚀 [Startup] Initializing House Muse SDXL + ControlNet + LoRA on ZeroGPU...")
t_start = time.time()

# 1. Depth-Anything estimator for extracting room geometry
print("[Startup] Loading Depth-Anything preprocessor...")
depth_estimator = pipeline(
    "depth-estimation",
    model="LiheYoung/depth-anything-small-hf",
    device="cuda",
)

# 2. ControlNet Depth model for SDXL
print("[Startup] Loading SDXL Depth ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16,
)

# 3. SDXL VAE with FP16 NaN fix
print("[Startup] Loading SDXL FP16 VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

# 4. SDXL Base Pipeline
print("[Startup] Loading SDXL Base Pipeline...")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

# Use fast DPM-Solver++ scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
)
pipe.vae.enable_tiling()

# 4b. SDXL ControlNet Img2Img Pipeline for Iterative Refinements
print("[Startup] Initializing SDXL ControlNet Img2Img Pipeline...")
try:
    pipe_refine = StableDiffusionXLControlNetImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        controlnet=pipe.controlnet,
        scheduler=pipe.scheduler,
    )
    pipe_refine.vae.enable_tiling()
    print("  ✓ Refine Img2Img pipeline initialized successfully (shared weights)")
except Exception as e:
    print(f"  ✗ Could not initialize StableDiffusionXLControlNetImg2ImgPipeline: {e}")
    pipe_refine = None

# 5. Pre-load all 9 LoRA adapters into memory

print(f"[Startup] Pre-loading all 9 LoRA adapters from: {LORA_DIR.resolve()}...")
loaded_adapters = set()

# Check if optional HuggingFace Model Repo is provided via environment variable
hf_lora_repo = os.environ.get("HF_LORA_REPO", "").strip()

for adapter_name, filename in LORA_FILES.items():
    local_path = LORA_DIR / filename
    if local_path.exists():
        try:
            pipe.load_lora_weights(str(local_path), adapter_name=adapter_name)
            loaded_adapters.add(adapter_name)
            print(f"  ✓ Loaded LoRA: '{adapter_name}' ({filename})")
        except Exception as e:
            print(f"  ✗ Error loading '{adapter_name}': {e}")
    elif hf_lora_repo:
        # Fallback to downloading from Hugging Face Hub model repo
        try:
            from huggingface_hub import hf_hub_download
            cached_file = hf_hub_download(repo_id=hf_lora_repo, filename=filename)
            pipe.load_lora_weights(cached_file, adapter_name=adapter_name)
            loaded_adapters.add(adapter_name)
            print(f"  ✓ Downloaded & loaded LoRA from Hub: '{adapter_name}'")
        except Exception as e:
            print(f"  ✗ Failed to download LoRA '{adapter_name}' from Hub: {e}")
    else:
        print(f"  ✗ LoRA file not found: {local_path}")

print(f"✨ [Startup Ready] Pipeline and {len(loaded_adapters)} LoRA adapters loaded in {round(time.time() - t_start, 2)}s.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def prepare_image(raw_image: Image.Image, target_max: int = 1024) -> Image.Image:
    """Resize input photo preserving aspect ratio, snapped to multiples of 64."""
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


def compute_depth(input_image: Image.Image) -> Image.Image:
    """Extract depth map using Depth-Anything."""
    depth_output = depth_estimator(input_image)["depth"]
    depth_rgb = depth_output.convert("RGB")
    return depth_rgb.resize(input_image.size, Image.Resampling.LANCZOS)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ZeroGPU GENERATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

@spaces.GPU(duration=60)
def generate_redesign(
    image: Image.Image,
    style: str,
    prompt: str = "",
    negative_prompt: str = "",
    controlnet_scale: float = 0.55,
    lora_scale: float = 0.80,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = -1,
):
    """
    ZeroGPU-accelerated room redesign function.
    Can be called directly from Gradio UI or programmatically via gradio_client.
    """
    if image is None:
        return None, "⚠️ Please upload a room photo first."

    t0 = time.time()

    # 1. Preprocess input & extract room geometry
    processed_image = prepare_image(image.convert("RGB"))
    depth_map = compute_depth(processed_image)

    # 2. Select LoRA adapter
    normalized_style = style.lower().strip()
    adapter_key = STYLE_TO_ADAPTER.get(normalized_style, normalized_style)

    if adapter_key in loaded_adapters:
        pipe.set_adapters([adapter_key], adapter_weights=[lora_scale])
        applied_note = f"Applied LoRA: '{adapter_key}' (weight={lora_scale})"
    else:
        pipe.disable_lora()
        applied_note = f"Base SDXL (No LoRA adapter active for '{style}')"

    # 3. Formulate prompts
    if prompt and prompt.strip():
        final_prompt = prompt.strip()
    else:
        final_prompt = DEFAULT_STYLE_PROMPTS.get(
            adapter_key,
            f"high-end interior design photograph of a {normalized_style} redesigned room, completely refurnished with new {normalized_style} furniture, new seating, new lighting fixtures, keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows, elegant, photorealistic, 8k"
        )

    final_negative = negative_prompt.strip() if negative_prompt and negative_prompt.strip() else DEFAULT_NEGATIVE_PROMPT

    # 4. Seed
    effective_seed = seed if seed >= 0 else int(time.time() * 1000) % (2**31)
    generator = torch.Generator(device="cuda").manual_seed(effective_seed)

    # 5. Run Diffusion
    result_image = pipe(
        prompt=final_prompt,
        negative_prompt=final_negative,
        image=depth_map,
        controlnet_conditioning_scale=controlnet_scale,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    duration = round(time.time() - t0, 2)
    status_msg = (
        f"✅ **Generation Complete** ({duration}s)\n\n"
        f"- **Style**: `{style}`\n"
        f"- **Adapter**: `{applied_note}`\n"
        f"- **Seed**: `{effective_seed}`\n"
        f"- **Prompt**: *{final_prompt}*"
    )

    return result_image, status_msg


@spaces.GPU(duration=60)
def generate_refine(
    image: Image.Image,
    style: str = "contemporary_indian",
    instruction: str = "",
    original_image: Optional[Image.Image] = None,
    prompt: str = "",
    negative_prompt: str = "",
    strength: float = 0.35,
    controlnet_scale: float = 0.55,
    lora_scale: float = 0.80,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = -1,
):
    """
    ZeroGPU room refinement function.
    Edits a previously generated room image using img2img with low strength (~0.3-0.4),
    preserving the room's established structure from the original photo's depth map
    and the previous style aesthetic while incorporating the text instruction.
    """
    if image is None:
        return None, "⚠️ Please provide the previous generated image to refine."

    t0 = time.time()

    # 1. Preprocess init image (most recent generated image)
    init_image = prepare_image(image.convert("RGB"))

    # 2. Extract geometry/depth map: prefer original room photo if provided to lock structure
    if original_image is not None:
        geometry_ref = prepare_image(original_image.convert("RGB"))
        depth_map = compute_depth(geometry_ref)
    else:
        depth_map = compute_depth(init_image)

    # 3. Select LoRA adapter
    normalized_style = style.lower().strip()
    adapter_key = STYLE_TO_ADAPTER.get(normalized_style, normalized_style)

    active_pipe = pipe_refine if pipe_refine is not None else pipe
    if adapter_key in loaded_adapters:
        active_pipe.set_adapters([adapter_key], adapter_weights=[lora_scale])
        applied_note = f"Applied LoRA: '{adapter_key}' (weight={lora_scale})"
    else:
        active_pipe.disable_lora()
        applied_note = f"Base SDXL (No LoRA adapter active for '{style}')"

    # 4. Formulate refined prompt: base style prompt + instruction
    if prompt and prompt.strip():
        base_prompt = prompt.strip()
    else:
        base_prompt = DEFAULT_STYLE_PROMPTS.get(
            adapter_key,
            f"high-end interior design photograph of a {normalized_style} redesigned room, completely refurnished with new {normalized_style} furniture, new seating, new lighting fixtures, keep all existing windows, doors and natural light openings exactly in place, do not cover or replace windows, elegant, photorealistic, 8k"
        )

    if instruction and instruction.strip():
        clean_instruction = instruction.strip()
        if clean_instruction.lower() not in base_prompt.lower():
            final_prompt = f"{base_prompt}, {clean_instruction}"
        else:
            final_prompt = base_prompt
    else:
        final_prompt = base_prompt

    final_negative = negative_prompt.strip() if negative_prompt and negative_prompt.strip() else DEFAULT_NEGATIVE_PROMPT

    # 5. Seed
    effective_seed = seed if seed >= 0 else int(time.time() * 1000) % (2**31)
    generator = torch.Generator(device="cuda").manual_seed(effective_seed)

    # 6. Run Diffusion (img2img with lower strength ~0.3-0.4)
    effective_strength = float(strength) if strength is not None else 0.35
    effective_strength = max(0.1, min(0.6, effective_strength))

    if pipe_refine is not None:
        result_image = pipe_refine(
            prompt=final_prompt,
            negative_prompt=final_negative,
            image=init_image,
            control_image=depth_map,
            strength=effective_strength,
            controlnet_conditioning_scale=float(controlnet_scale),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]
    else:
        result_image = pipe(
            prompt=final_prompt,
            negative_prompt=final_negative,
            image=depth_map,
            controlnet_conditioning_scale=float(controlnet_scale),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]

    duration = round(time.time() - t0, 2)
    status_msg = (
        f"✅ **Refinement Complete** ({duration}s)\n\n"
        f"- **Instruction**: *{instruction}*\n"
        f"- **Strength**: `{effective_strength}`\n"
        f"- **Style**: `{style}`\n"
        f"- **Adapter**: `{applied_note}`\n"
        f"- **Seed**: `{effective_seed}`\n"
        f"- **Prompt**: *{final_prompt}*"
    )

    return result_image, status_msg


# ══════════════════════════════════════════════════════════════════════════════
# 5. GRADIO INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

custom_css = """
#main-container { max-width: 1200px; margin: 0 auto; padding: 10px; }
.generate-btn { background: #E2C89A !important; color: #1E1E1E !important; font-weight: 600 !important; }
"""

with gr.Blocks(title="House Muse — AI Room Redesign", css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown(
            """
            # 🏠 House Muse — AI Interior Redesign & Refinement
            **ZeroGPU Serverless Studio**: Keep your room's architectural layout (walls, windows, geometry)
            while re-imagining or iteratively refining the interior in one of 9 curated styles using **SDXL + Depth ControlNet + LoRA**.
            """
        )

        with gr.Tabs():
            with gr.Tab("Redesign Room"):
                with gr.Row():
                    with gr.Column(scale=5):
                        image_input = gr.Image(
                            type="pil",
                            label="1. Upload Room Photograph",
                            sources=["upload", "clipboard"],
                        )
                        style_dropdown = gr.Dropdown(
                            choices=STYLE_CHOICES,
                            value="contemporary_indian",
                            label="2. Select Interior Style",
                        )

                        with gr.Accordion("⚙️ Advanced Generation Settings", open=False):
                            prompt_input = gr.Textbox(
                                label="Custom Prompt (optional)",
                                placeholder="Leave blank to use tailored style trigger prompt...",
                                lines=2,
                            )
                            negative_prompt_input = gr.Textbox(
                                label="Negative Prompt",
                                value=DEFAULT_NEGATIVE_PROMPT,
                                lines=2,
                            )
                            with gr.Row():
                                controlnet_slider = gr.Slider(
                                    minimum=0.2, maximum=1.2, value=0.55, step=0.05,
                                    label="ControlNet Depth (Geometry Lock)",
                                    info="Higher = stricter room shape preservation (0.5-0.6 recommended for refurnishing)"
                                )
                                lora_slider = gr.Slider(
                                    minimum=0.2, maximum=1.2, value=0.80, step=0.05,
                                    label="LoRA Style Strength",
                                    info="Higher = stronger style aesthetic"
                                )
                            with gr.Row():
                                steps_slider = gr.Slider(
                                    minimum=20, maximum=45, value=30, step=1,
                                    label="Denoising Steps"
                                )
                                guidance_slider = gr.Slider(
                                    minimum=3.0, maximum=12.0, value=7.5, step=0.5,
                                    label="Guidance Scale (CFG)"
                                )
                            seed_input = gr.Number(
                                value=-1,
                                label="Seed (-1 for random)",
                                precision=0,
                            )

                        submit_btn = gr.Button("Redesign Room ✨", variant="primary", elem_classes=["generate-btn"])

                    with gr.Column(scale=5):
                        image_output = gr.Image(type="pil", label="Redesigned Room Output")
                        status_output = gr.Markdown(label="Generation Details")

                submit_btn.click(
                    fn=generate_redesign,
                    inputs=[
                        image_input,
                        style_dropdown,
                        prompt_input,
                        negative_prompt_input,
                        controlnet_slider,
                        lora_slider,
                        steps_slider,
                        guidance_slider,
                        seed_input,
                    ],
                    outputs=[image_output, status_output],
                    api_name="redesign",  # <-- Programmatic API endpoint: /redesign
                )

            with gr.Tab("Refine Design"):
                with gr.Row():
                    with gr.Column(scale=5):
                        refine_image_input = gr.Image(
                            type="pil",
                            label="1. Most Recent Generated Image (Init Image)",
                            sources=["upload", "clipboard"],
                        )
                        refine_instruction_input = gr.Textbox(
                            label="2. Refinement Instruction",
                            placeholder="e.g. make the bed smaller, add a reading chair",
                            lines=2,
                        )
                        refine_orig_image_input = gr.Image(
                            type="pil",
                            label="3. Original Room Photo (Geometry Lock, Optional)",
                            sources=["upload", "clipboard"],
                        )
                        refine_style_dropdown = gr.Dropdown(
                            choices=STYLE_CHOICES,
                            value="contemporary_indian",
                            label="4. Style",
                        )

                        with gr.Accordion("⚙️ Refinement Settings", open=False):
                            refine_prompt_input = gr.Textbox(
                                label="Base Prompt (optional)",
                                placeholder="Leave blank to use base style prompt...",
                                lines=2,
                            )
                            refine_negative_prompt_input = gr.Textbox(
                                label="Negative Prompt",
                                value=DEFAULT_NEGATIVE_PROMPT,
                                lines=2,
                            )
                            refine_strength_slider = gr.Slider(
                                minimum=0.1, maximum=0.6, value=0.35, step=0.05,
                                label="Denoising Strength (0.3-0.4 recommended for editing)",
                                info="Lower = subtle edits preserving existing image; Higher = more re-rendering"
                            )
                            with gr.Row():
                                refine_controlnet_slider = gr.Slider(
                                    minimum=0.2, maximum=1.2, value=0.55, step=0.05,
                                    label="ControlNet Depth Scale",
                                )
                                refine_lora_slider = gr.Slider(
                                    minimum=0.2, maximum=1.2, value=0.80, step=0.05,
                                    label="LoRA Scale",
                                )
                            with gr.Row():
                                refine_steps_slider = gr.Slider(
                                    minimum=20, maximum=45, value=30, step=1,
                                    label="Steps"
                                )
                                refine_guidance_slider = gr.Slider(
                                    minimum=3.0, maximum=12.0, value=7.5, step=0.5,
                                    label="CFG Scale"
                                )
                            refine_seed_input = gr.Number(
                                value=-1,
                                label="Seed (-1 for random)",
                                precision=0,
                            )

                        refine_submit_btn = gr.Button("Refine Design ✏️", variant="primary", elem_classes=["generate-btn"])

                    with gr.Column(scale=5):
                        refine_image_output = gr.Image(type="pil", label="Refined Room Output")
                        refine_status_output = gr.Markdown(label="Refinement Details")

                refine_submit_btn.click(
                    fn=generate_refine,
                    inputs=[
                        refine_image_input,
                        refine_style_dropdown,
                        refine_instruction_input,
                        refine_orig_image_input,
                        refine_prompt_input,
                        refine_negative_prompt_input,
                        refine_strength_slider,
                        refine_controlnet_slider,
                        refine_lora_slider,
                        refine_steps_slider,
                        refine_guidance_slider,
                        refine_seed_input,
                    ],
                    outputs=[refine_image_output, refine_status_output],
                    api_name="refine",  # <-- Programmatic API endpoint: /refine
                )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

