"""
modules/image_cache.py — Image Memory, Depth Caching & Quota Protection

Handles:
  1. Saving generated/refined images in gitignored ml/cache/ folder
  2. Keeping only the last 10 images per project (FIFO eviction)
  3. Pre-computing and caching CPU depth maps per project to avoid GPU work
  4. Hashing refine requests (image_id + instruction + settings) to avoid duplicate GPU calls
"""

import os
import io
import json
import uuid
import hashlib
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_BASE_DIR, "cache")
INDEX_FILE = os.path.join(CACHE_DIR, "cache_index.json")
REFINE_CACHE_FILE = os.path.join(CACHE_DIR, "refine_hash_cache.json")

os.makedirs(CACHE_DIR, exist_ok=True)


def _load_json(file_path: str) -> dict:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(file_path: str, data: dict):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        print(f"[ImageCache] Failed saving index {file_path}: {exc}")


def _enforce_project_limit(project_id: str, max_items: int = 10):
    """Retains only the last max_items images for project_id, deleting older ones."""
    if not project_id:
        return
    index = _load_json(INDEX_FILE)
    proj_key = str(project_id)
    items = index.get(proj_key, [])

    if len(items) > max_items:
        # Items to evict: oldest first
        evict_count = len(items) - max_items
        to_evict = items[:evict_count]
        remaining = items[evict_count:]

        for img_id in to_evict:
            img_path = os.path.join(CACHE_DIR, f"{img_id}.jpg")
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                    print(f"[ImageCache] Evicted old cached image {img_id} for project {project_id}")
                except OSError:
                    pass

        index[proj_key] = remaining
        _save_json(INDEX_FILE, index)


def save_cached_image(
    arg1: Any,
    arg2: Any = None,
    image_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Saves image into CACHE_DIR as {image_id}.jpg.
    Accepts either (project_id, image_data) or (image_data, project_id).
    Returns (image_id, image_url).
    """
    if isinstance(arg1, (bytes, bytearray, Image.Image)) or (
        isinstance(arg1, str) and (len(arg1) > 100 or os.path.exists(arg1) or arg1.startswith("data:"))
    ):
        image_data = arg1
        project_id = arg2
    else:
        project_id = arg1
        image_data = arg2

    if not image_id:
        image_id = uuid.uuid4().hex[:10]

    file_name = f"{image_id}.jpg"
    file_path = os.path.join(CACHE_DIR, file_name)

    # Convert image_data into JPEG bytes and write
    if isinstance(image_data, (bytes, bytearray)):
        with open(file_path, "wb") as f:
            f.write(image_data)
    elif isinstance(image_data, Image.Image):
        image_data.convert("RGB").save(file_path, format="JPEG", quality=92)
    elif isinstance(image_data, str):
        if os.path.exists(image_data):
            with open(image_data, "rb") as src, open(file_path, "wb") as dst:
                dst.write(src.read())
        else:
            import base64
            clean_b64 = image_data.split(",", 1)[1] if "," in image_data else image_data
            missing_padding = len(clean_b64) % 4
            if missing_padding:
                clean_b64 += "=" * (4 - missing_padding)
            try:
                raw_bytes = base64.b64decode(clean_b64)
            except Exception:
                raw_bytes = image_data.encode("utf-8")
            with open(file_path, "wb") as f:
                f.write(raw_bytes)
    else:
        raise ValueError(f"Unsupported image_data type: {type(image_data)}")

    # Update project index
    if project_id:
        proj_key = str(project_id)
        index = _load_json(INDEX_FILE)
        items = index.get(proj_key, [])
        if image_id not in items:
            items.append(image_id)
        index[proj_key] = items
        _save_json(INDEX_FILE, index)
        _enforce_project_limit(proj_key, max_items=10)

    image_url = f"/api/image/{image_id}"
    return image_id, image_url


def get_cached_image_path(image_id: str) -> Optional[str]:
    """Returns local file path if cached image exists, else None."""
    if not image_id:
        return None
    file_path = os.path.join(CACHE_DIR, f"{image_id}.jpg")
    if os.path.exists(file_path):
        return file_path
    return None


def get_cached_image(image_id: str) -> Optional[Tuple[bytes, str]]:
    """Returns (image_bytes, mime_type) if cached image exists, else None."""
    path = get_cached_image_path(image_id)
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return f.read(), "image/jpeg"
        except Exception:
            return None
    return None


# ─── CPU Depth Map Caching ───────────────────────────────────────────────────

def get_cached_depth_path(project_id: Any) -> Optional[str]:
    """Returns path to cached precomputed depth map for project_id if present."""
    if not project_id:
        return None
    path = os.path.join(CACHE_DIR, f"depth_proj_{project_id}.jpg")
    if os.path.exists(path):
        return path
    return None


def compute_and_cache_depth(project_id: Any, image_source: Any) -> Optional[str]:
    """
    Computes a structural depth/geometry map on CPU once per project and caches it.
    Reuses cached depth map if already computed.
    """
    if not project_id or not image_source:
        return None

    cached = get_cached_depth_path(project_id)
    if cached:
        return cached

    depth_path = os.path.join(CACHE_DIR, f"depth_proj_{project_id}.jpg")

    try:
        # Load image into numpy array
        if isinstance(image_source, (bytes, bytearray)):
            nparr = np.frombuffer(image_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_source, str) and os.path.exists(image_source):
            img = cv2.imread(image_source)
        elif isinstance(image_source, Image.Image):
            img = cv2.cvtColor(np.array(image_source.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            return None

        if img is None:
            return None

        # Resize to max 768px for fast CPU processing and quota efficiency
        h, w = img.shape[:2]
        max_side = 768
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # CPU Structural geometry depth approximation:
        # 1. Bilateral filter preserves sharp wall & window boundaries while smoothing textures
        filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        # 2. CLAHE local contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        norm_gray = clahe.apply(gray)

        # 3. Morphological gradient emphasizes architectural planes (walls, openings, floor edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.morphologyEx(norm_gray, cv2.MORPH_GRADIENT, kernel)

        # Combine luminance depth gradient with structural edges
        depth_est = cv2.addWeighted(norm_gray, 0.75, grad, 0.25, 0)
        depth_rgb = cv2.cvtColor(depth_est, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(depth_path, depth_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"[ImageCache] Precomputed & cached CPU depth map for project {project_id} -> {depth_path}")
        return depth_path
    except Exception as exc:
        print(f"[ImageCache] Failed computing CPU depth map for project {project_id}: {exc}")
        return None


# ─── Refine Hash Caching ─────────────────────────────────────────────────────

def make_refine_hash(image_id: str, instruction: str, settings: Optional[dict] = None) -> str:
    settings_str = json.dumps(settings or {}, sort_keys=True)
    key_str = f"{image_id}:{instruction.strip().lower()}:{settings_str}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]


def get_cached_refine_result(
    image_id_or_hash: str,
    instruction: Optional[str] = None,
    settings: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    if instruction is not None:
        cache_hash = make_refine_hash(image_id_or_hash, instruction, settings)
    else:
        cache_hash = image_id_or_hash
    cache = _load_json(REFINE_CACHE_FILE)
    entry = cache.get(cache_hash)
    if entry:
        img_id = entry.get("image_id")
        if img_id and get_cached_image_path(img_id):
            return entry
    return None


def save_cached_refine_result(
    image_id_or_hash: str,
    instruction_or_result: Any,
    settings: Optional[dict] = None,
    result_dict: Optional[Dict[str, Any]] = None,
):
    if result_dict is not None:
        cache_hash = make_refine_hash(image_id_or_hash, instruction_or_result, settings)
        data = result_dict
    elif isinstance(instruction_or_result, dict):
        cache_hash = image_id_or_hash
        data = instruction_or_result
    else:
        return
    cache = _load_json(REFINE_CACHE_FILE)
    cache[cache_hash] = data
    _save_json(REFINE_CACHE_FILE, cache)
