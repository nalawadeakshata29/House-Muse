"""
app.py — House Muse Backend API
Full stack: YOLOv8 + SAM + CLIP + Places365
"""

import os, json, traceback
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from modules.image_processor        import process_image
from modules.color_palette          import generate_palette
from modules.space_analyzer         import analyse_space
from modules.recommendation_engine  import recommend, recommendation_to_dict, get_style_palette
from modules.image_generator        import (
    generate_room_image,
    generate_multiple_variants,
    refine_room_image,
    get_base_style_prompt,
)
from modules.caption_engine         import get_generation_prompt
from modules.image_cache            import (
    save_cached_image,
    get_cached_image,
    compute_and_cache_depth,
    get_cached_depth_path,
)
import numpy as np

app = Flask(__name__)
CORS(app, origins=["http://localhost:5500","http://127.0.0.1:5500",
                   "http://localhost:3000","null"])
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

os.makedirs("uploads", exist_ok=True)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "House Muse — Full AI Stack"})


@app.route("/api/image/<image_id>", methods=["GET"])
def get_cached_image_route(image_id):
    cached = get_cached_image(image_id)
    if not cached:
        return jsonify({"error": "image_expired", "code": "image_expired"}), 404
    data, mime = cached
    return Response(data, mimetype=mime)


@app.route("/api/analyse", methods=["POST"])
def analyse():
    try:
        room_type  = request.form.get("room_type",  "living_room")
        length     = float(request.form.get("length", 12))
        width      = float(request.form.get("width",  10))
        height     = float(request.form.get("height",  9))
        styles_raw = request.form.get("styles", '["Japandi"]')
        styles     = json.loads(styles_raw) if isinstance(styles_raw, str) else styles_raw
        budget     = float(request.form.get("budget", 50000))
        scope      = request.form.get("scope",    "both")
        num_rooms  = int(request.form.get("num_rooms", 1))
        harmony    = request.form.get("harmony",  "analogous")

        # ── 1. Space analysis ──────────────────────────────────────────────
        space = analyse_space(length, width, height, room_type)

        # ── 2. Full image pipeline ─────────────────────────────────────────
        image_file = request.files.get("image")

        if image_file and image_file.filename:
            img_result     = process_image(image_file.read(),
                                           provided_room_type=room_type)
            dominant_colors = img_result["dominant_colors"]
            lighting        = img_result["lighting"]["condition"]
            detected_room   = img_result["room_type"]
            detections      = img_result.get("detections",    [])
            object_colors   = img_result.get("object_colors", [])
            object_palette  = img_result.get("object_palette",{})
            style_sims      = img_result.get("style_similarities", {})
            clip_features   = np.array(img_result.get("cnn_features", []), dtype=np.float32)
            room_method     = img_result.get("room_type_method", "user_provided")
            lighting_full   = img_result["lighting"]
        else:
            dominant_colors = _default_colors(room_type)
            lighting        = "bright"
            detected_room   = room_type
            detections      = []
            object_colors   = []
            object_palette  = {}
            style_sims      = {}
            clip_features   = None
            room_method     = "user_provided"
            lighting_full   = {"condition":"bright","mean_brightness":128,"color_temperature":"neutral"}

        # ── 3. Style-driven palette (style is PRIMARY driver) ──────────────
        # Style selection overrides object-detected palette
        # so Vintage Indian always gives earthy ochres, Art Deco gives gold+navy etc.
        style_palette = get_style_palette(styles, dominant_colors, lighting)

        palette = generate_palette(
            dominant_colors = dominant_colors,
            room_size       = space.size_category,
            ceiling_height  = space.ceiling_class,
            lighting        = lighting,
            style           = styles[0] if styles else "Japandi",
            harmony         = harmony,
        )

        # Override with style-driven colours
        palette.update({
            "wall_color":          style_palette["wall_color"],
            "accent_color":        style_palette["accent_color"],
            "ceiling_color":       style_palette["ceiling_color"],
            "object_palette_note": style_palette["object_palette_note"],
            "anchor_object":       style_palette.get("anchor_object"),
            "palette_strategy":    style_palette["palette_strategy"],
            "detected_style":      style_palette["detected_style"],
            "style_similarities":  style_sims,
            "colour_mood":         style_palette.get("colour_mood",""),
        })

        # ── 4. Recommendations ─────────────────────────────────────────────
        wall_area = 2 * (length + width) * height
        reco = recommend(
            dominant_colors = dominant_colors,
            room_type       = detected_room,
            size_category   = space.size_category,
            styles          = styles,
            total_budget    = budget,
            scope           = scope,
            wall_area_sqft  = wall_area,
            image_features  = clip_features,
            object_colors   = object_colors,
            style_sims      = style_sims,
        )

        # ── 5. Response ────────────────────────────────────────────────────
        return jsonify({
            "success": True,
            "space": {
                "area":                  space.area,
                "volume":                space.volume,
                "size_category":         space.size_category,
                "ceiling_class":         space.ceiling_class,
                "shape_hint":            space.shape_hint,
                "color_guidance":        space.color_guidance,
                "furniture_scale":       space.furniture_scale,
                "pattern_scale":         space.pattern_scale,
                "lighting_suggestions":  space.lighting_suggestions,
                "layout_tips":           space.layout_tips,
                "wallpaper_suitability": space.wallpaper_suitability,
            },
            "palette": {k:v for k,v in palette.items() if k != "cnn_features"},
            "image_analysis": {
                "dominant_colors":   dominant_colors,
                "lighting":          lighting_full,
                "detected_room":     detected_room,
                "room_type_method":  room_method,
                "detections":        detections,
                "object_colors":     object_colors,
                "style_similarities":style_sims,
            },
            "object_palette":  style_palette,
            "recommendations": recommendation_to_dict(reco),
            "meta": {
                "num_rooms": num_rooms, "scope": scope,
                "styles": styles, "budget": budget,
                "wall_area": round(wall_area, 1),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/palette", methods=["POST"])
def palette_only():
    try:
        data      = request.get_json(force=True)
        room_type = data.get("room_type","living_room")
        length    = float(data.get("length",12))
        width     = float(data.get("width", 10))
        height    = float(data.get("height", 9))
        styles    = data.get("styles",["Japandi"])
        harmony   = data.get("harmony","analogous")

        space   = analyse_space(length, width, height, room_type)
        colors  = _default_colors(room_type)
        palette = generate_palette(colors, space.size_category,
                                   space.ceiling_class, "bright",
                                   styles[0] if styles else "Japandi", harmony)

        return jsonify({"success":True,"palette":palette,"space":{
            "size_category":  space.size_category,
            "ceiling_class":  space.ceiling_class,
            "color_guidance": space.color_guidance,
        }})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500


@app.route("/api/generate-image", methods=["POST"])
def generate_image():
    """
    Full pipeline:
      1. BLIP encoder-decoder → caption from room image
      2. Prompt builder → rich SD prompt from caption + style + palette
      3. Hugging Face ZeroGPU Space (SDXL + ControlNet + LoRA) → styled room image.
         On failure returns proper JSON error (429 quota, 503 Space asleep, 502 other).
    """
    try:
        # Parse — supports both JSON and multipart
        is_json = request.content_type and "application/json" in request.content_type

        if is_json:
            data             = request.get_json(force=True)
            image_bytes      = None
            project_id       = data.get("project_id")
            room_type        = data.get("room_type",       "living_room")
            styles           = data.get("styles",          ["Japandi"])
            wall_color       = data.get("wall_color",      "#E2C89A")
            accent_color     = data.get("accent_color",    "#8A9E85")
            dominant_colors  = data.get("dominant_colors", [])
            size_category    = data.get("size_category",   "medium")
            lighting         = data.get("lighting",        "bright")
            detections       = data.get("detections",      [])
            n_variants       = int(data.get("n_variants",  1))
        else:
            image_file       = request.files.get("image")
            image_bytes      = image_file.read() if image_file else None
            project_id       = request.form.get("project_id")
            room_type        = request.form.get("room_type",       "living_room")
            styles_raw       = request.form.get("styles") or request.form.get("style", '["Japandi"]')
            if isinstance(styles_raw, str):
                try:
                    styles   = json.loads(styles_raw)
                except Exception:
                    styles   = [styles_raw]
            else:
                styles       = styles_raw or ["Japandi"]
            if not isinstance(styles, list):
                styles       = [styles]

            wall_color       = request.form.get("wall_color",      "#E2C89A")
            accent_color     = request.form.get("accent_color",    "#8A9E85")

            dominant_colors_raw = request.form.get("dominant_colors", "[]")
            try:
                dominant_colors = json.loads(dominant_colors_raw) if isinstance(dominant_colors_raw, str) else dominant_colors_raw
            except Exception:
                dominant_colors = []

            size_category    = request.form.get("size_category",   "medium")
            lighting         = request.form.get("lighting",        "bright")

            detections_raw   = request.form.get("detections", "[]")
            try:
                detections   = json.loads(detections_raw) if isinstance(detections_raw, str) else detections_raw
            except Exception:
                detections   = []

            n_variants       = int(request.form.get("n_variants",  1))

        # Precompute and cache depth map once on CPU on Flask side
        if image_bytes and project_id:
            try:
                compute_and_cache_depth(project_id, image_bytes)
            except Exception as depth_err:
                print(f"[Generate] Depth caching warning: {depth_err}")

        # ── Step 1+2: BLIP caption → prompt ────────────────────────────────
        prompt_data = get_generation_prompt(
            image_bytes      = image_bytes,
            room_type        = room_type,
            styles           = styles,
            wall_color       = wall_color,
            accent_color     = accent_color,
            size_category    = size_category,
            lighting         = lighting,
            detected_objects = detections,
            dominant_colors  = dominant_colors,
        )

        positive = prompt_data["positive_prompt"]
        negative = prompt_data["negative_prompt"]
        caption  = prompt_data["caption"]
        method   = prompt_data["method"]

        print(f"[Generate] Method: {method}")
        if caption:
            print(f"[Generate] BLIP caption: {caption}")

        # ── Step 3: Hugging Face Space generation ─────────────────────────
        chosen_style = styles[0] if styles else "contemporary_indian"
        if n_variants == 1:
            result   = generate_room_image(positive, negative, image_input=image_bytes, style=chosen_style, room_type=room_type)
            image_id = None
            image_url = result.get("image_url")
            if result.get("success"):
                img_data = result.get("image_b64") or result.get("image_path")
                if img_data:
                    image_id, cached_url = save_cached_image(project_id or "default", img_data)
                    image_url = cached_url
                    result["image_id"] = image_id
                    result["image_url"] = image_url

            variants = [{
                **result,
                "variant":       1,
                "variant_label": "Styled Room",
                "caption":       caption,
                "prompt_method": method,
                "image_id":      image_id,
                "image_url":     image_url,
            }]
        else:
            labels = ["Full Room View", "Detail View", "Evening Mood"]
            notes  = [
                "wide angle view showing full room",
                "close up corner detail showing furniture",
                "evening warm lighting cosy atmosphere",
            ]
            variants = []
            for i in range(min(n_variants, 3)):
                extra_pos = f"{positive}, {notes[i]}"
                r = generate_room_image(extra_pos, negative, image_input=image_bytes, style=chosen_style, room_type=room_type)
                v_img_id = None
                v_img_url = r.get("image_url")
                if r.get("success"):
                    img_data = r.get("image_b64") or r.get("image_path")
                    if img_data:
                        v_img_id, v_img_url = save_cached_image(project_id or "default", img_data)
                        r["image_id"] = v_img_id
                        r["image_url"] = v_img_url
                variants.append({
                    **r,
                    "variant":       i + 1,
                    "variant_label": labels[i],
                    "caption":       caption,
                    "prompt_method": method,
                    "image_id":      v_img_id,
                    "image_url":     v_img_url,
                })
                import time; time.sleep(1)

        success = any(v.get("success") for v in variants)
        primary = variants[0] if variants else {}
        if not success:
            status_code = primary.get("status_code", 502)
            error_msg = primary.get("error", "Generation failed.")
            return jsonify({
                "success": False,
                "error": error_msg,
                "caption": caption,
                "method": method,
                "variants": variants,
            }), status_code

        return jsonify({
            "success":   True,
            "image_id":  primary.get("image_id"),
            "image_url": primary.get("image_url"),
            "image_b64": primary.get("image_b64"),
            "caption":   caption,
            "method":    method,
            "variants":  variants,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "variants": []}), 500


@app.route("/api/refine-image", methods=["POST"])
def refine_image():
    """
    Refinement pipeline:
      1. Receives currentImageId (or init image data as fallback)
      2. Uses cached depth map from original photo for ControlNet layout preservation
      3. Refines image with strength ~0.50 and steps 25
      4. Returns {image_id, image_url, image_b64, ...}
    """
    try:
        is_json = request.content_type and "application/json" in request.content_type

        if is_json:
            data                 = request.get_json(force=True)
            project_id           = data.get("project_id") or "default"
            image_id             = data.get("image_id") or data.get("current_image_id")
            image_input          = data.get("image") or data.get("init_image")
            original_image_input = data.get("original_image")
            instruction          = data.get("instruction", "").strip()
            room_type            = data.get("room_type", "living_room")
            styles               = data.get("styles") or [data.get("style", "contemporary_indian")]
            strength_raw         = data.get("strength")
            strength             = float(strength_raw) if strength_raw is not None else None
            custom_prompt        = data.get("prompt", "").strip()
        else:
            project_id           = request.form.get("project_id") or "default"
            image_id             = request.form.get("image_id") or request.form.get("current_image_id")
            image_file           = request.files.get("image") or request.files.get("init_image")
            image_input          = image_file.read() if image_file else request.form.get("image") or request.form.get("init_image")
            orig_file            = request.files.get("original_image")
            original_image_input = orig_file.read() if orig_file else request.form.get("original_image")
            instruction          = request.form.get("instruction", "").strip()
            room_type            = request.form.get("room_type", "living_room")
            styles_raw           = request.form.get("styles") or request.form.get("style", '["contemporary_indian"]')
            if isinstance(styles_raw, str):
                try:
                    styles       = json.loads(styles_raw)
                except Exception:
                    styles       = [styles_raw]
            else:
                styles           = styles_raw or ["contemporary_indian"]
            if not isinstance(styles, list):
                styles           = [styles]
            strength_raw         = request.form.get("strength")
            strength             = float(strength_raw) if strength_raw is not None else None
            custom_prompt        = request.form.get("prompt", "").strip()

        if not instruction:
            return jsonify({"success": False, "error": "No instruction provided for refinement"}), 400

        # Resolve init image:
        # First try image_id from cache
        init_image_data = None
        if image_id:
            cached_img = get_cached_image(image_id)
            if cached_img:
                init_image_data = cached_img[0]
            else:
                # Image expired in cache (e.g. server restarted or purged)
                if not image_input:
                    return jsonify({"error": "image_expired", "code": "image_expired", "success": False}), 404
                # Fallback init_image was provided by frontend
                new_id, _ = save_cached_image(project_id, image_input)
                image_id = new_id
                init_image_data = image_input
        elif image_input:
            new_id, _ = save_cached_image(project_id, image_input)
            image_id = new_id
            init_image_data = image_input
        else:
            return jsonify({"success": False, "error": "No init image or image_id provided for refinement"}), 400

        # Resolve ControlNet condition (cached depth map of original room photo)
        controlnet_depth = None
        if project_id:
            cached_depth = get_cached_depth_path(project_id)
            if cached_depth:
                controlnet_depth = cached_depth
            elif original_image_input:
                try:
                    controlnet_depth = compute_and_cache_depth(project_id, original_image_input)
                except Exception as e:
                    print(f"[Refine] Depth computation warning: {e}")
                    controlnet_depth = original_image_input
        if not controlnet_depth and original_image_input:
            controlnet_depth = original_image_input

        chosen_style = styles[0] if styles else "contemporary_indian"

        # Build prompt: base style prompt + instruction appended
        if custom_prompt:
            base_prompt = custom_prompt
        else:
            base_prompt = get_base_style_prompt(style=chosen_style, room_type=room_type)

        preserve_clause = "keep the room's walls, windows, floor and ceiling unchanged"
        instruction_with_preserve = instruction
        if preserve_clause.lower() not in instruction.lower():
            instruction_with_preserve = f"{instruction}, {preserve_clause}"

        if instruction_with_preserve.lower() not in base_prompt.lower():
            full_prompt = f"{base_prompt}, {instruction_with_preserve}"
        else:
            full_prompt = base_prompt

        print(f"[Refine] Style: {chosen_style}, Project: {project_id}, ImageId: {image_id}")
        print(f"[Refine] Prompt: {full_prompt}")

        result = refine_room_image(
            positive_prompt=full_prompt,
            negative_prompt="",
            init_image=init_image_data,
            original_image=controlnet_depth,
            instruction=instruction,
            style=chosen_style,
            strength=strength,
            room_type=room_type,
            image_id=image_id,
        )

        success = result.get("success", False)
        status_code = result.get("status_code", 200 if success else 502)

        if not success:
            return jsonify({
                "success": False,
                "error": result.get("error", "Refinement failed."),
                "prompt": result.get("prompt", full_prompt),
                "instruction": instruction,
                "variants": [],
            }), status_code

        # Save new image to project cache
        img_payload = result.get("image_b64") or result.get("image_path")
        new_image_id, new_image_url = save_cached_image(project_id, img_payload)
        img_b64 = result.get("image_b64")

        return jsonify({
            "success": True,
            "image_id": new_image_id,
            "image_url": new_image_url,
            "image_b64": img_b64,
            "instruction": instruction,
            "prompt": result.get("prompt", full_prompt),
            "error": None,
            "variants": [
                {
                    "variant": 1,
                    "variant_label": "Refined View",
                    "image_id": new_image_id,
                    "image_url": new_image_url,
                    "image_b64": img_b64,
                    "prompt": result.get("prompt", full_prompt),
                    "instruction": instruction,
                    "success": True,
                    "error": None,
                }
            ],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "variants": []}), 500



# ── Default colours when no image uploaded ─────────────────────────────────
_ROOM_DEFAULTS = {
    "living_room":  [{"hex":"#C4A882","rgb":[196,168,130],"percentage":40},{"hex":"#8A9E85","rgb":[138,158,133],"percentage":25},{"hex":"#E8D5A3","rgb":[232,213,163],"percentage":20},{"hex":"#3D2E26","rgb":[61,46,38],"percentage":10},{"hex":"#C67C52","rgb":[198,124,82],"percentage":5}],
    "bedroom":      [{"hex":"#D4C5B0","rgb":[212,197,176],"percentage":40},{"hex":"#8A9E85","rgb":[138,158,133],"percentage":25},{"hex":"#F5EFE0","rgb":[245,239,224],"percentage":20},{"hex":"#6B5344","rgb":[107,83,68],"percentage":10},{"hex":"#C4856A","rgb":[196,133,106],"percentage":5}],
    "kitchen":      [{"hex":"#E8E4D8","rgb":[232,228,216],"percentage":45},{"hex":"#B0A090","rgb":[176,160,144],"percentage":25},{"hex":"#F5F0E8","rgb":[245,240,232],"percentage":20},{"hex":"#6B5344","rgb":[107,83,68],"percentage":8},{"hex":"#C9A84C","rgb":[201,168,76],"percentage":2}],
    "dining_room":  [{"hex":"#B87333","rgb":[184,115,51],"percentage":35},{"hex":"#C4A882","rgb":[196,168,130],"percentage":30},{"hex":"#3D2E26","rgb":[61,46,38],"percentage":20},{"hex":"#E8D5A3","rgb":[232,213,163],"percentage":10},{"hex":"#8C5F6E","rgb":[140,95,110],"percentage":5}],
    "bathroom":     [{"hex":"#E0EEF0","rgb":[224,238,240],"percentage":40},{"hex":"#B0C8CC","rgb":[176,200,204],"percentage":30},{"hex":"#F5F5F0","rgb":[245,245,240],"percentage":20},{"hex":"#6B8A8C","rgb":[107,138,140],"percentage":8},{"hex":"#C4A882","rgb":[196,168,130],"percentage":2}],
    "study_room":   [{"hex":"#4A3728","rgb":[74,55,40],"percentage":35},{"hex":"#8A7A6A","rgb":[138,122,106],"percentage":25},{"hex":"#C4A882","rgb":[196,168,130],"percentage":20},{"hex":"#D4C5B0","rgb":[212,197,176],"percentage":15},{"hex":"#6B2737","rgb":[107,39,55],"percentage":5}],
}
def _default_colors(room_type):
    return _ROOM_DEFAULTS.get(room_type, _ROOM_DEFAULTS["living_room"])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  🏠  House Muse API  →  http://localhost:{port}\n")
    app.run(debug=True, host="0.0.0.0", port=port)
