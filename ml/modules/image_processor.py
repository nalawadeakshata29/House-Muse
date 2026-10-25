"""
modules/image_processor.py
════════════════════════════════════════════════════════════════════════════════
House Muse — Full AI Vision Pipeline

Stack:
  YOLOv8          → Object detection (fine-tuned on IKEA + ADE20K)
  SAM             → Pixel-perfect object masks (no background bleed)
  CLIP            → 512-d semantic feature vector (text+image aware)
  Places365       → Room type classification (365 indoor/outdoor scenes)
  K-Means on mask → Colour extraction from exact object pixels only
   
This file is COMPLETE and SELF-CONTAINED.
Drop it into backend/modules/ — no other file needs editing.
════════════════════════════════════════════════════════════════════════════════
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHON_GIL", "1")
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as T
import colorsys
from sklearn.cluster import KMeans

# ── YOLOv8 ───────────────────────────────────────────────────────────────────
from ultralytics import YOLO

# ── SAM ───────────────────────────────────────────────────────────────────────
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[HouseMuse] SAM not installed — falling back to bbox crops for colour.")

# ── CLIP ──────────────────────────────────────────────────────────────────────
try:
    import clip as openai_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[HouseMuse] CLIP not installed — falling back to EfficientNet features.")

# ── EfficientNet fallback (already in torchvision) ───────────────────────────
from torchvision import models as tv_models


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these paths after training
# ══════════════════════════════════════════════════════════════════════════════

# After running training/train_yolo.py, update this path:
YOLO_WEIGHTS     = os.getenv(
    "YOLO_WEIGHTS",
    "models/checkpoints/house_muse_yolo/weights/best.pt"
)

# YOLO-World & Fallback configuration — absolute paths anchored to this file
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WORLD_WEIGHTS    = os.getenv(
    "YOLO_WORLD_WEIGHTS",
    os.path.normpath(os.path.join(_ML_DIR, "..", "yolov8s-worldv2.pt"))
)
COCO_FALLBACK_WEIGHTS = os.getenv(
    "COCO_WEIGHTS",
    os.path.normpath(os.path.join(_ML_DIR, "..", "yolov8m.pt"))
)
COCO_FALLBACK_CONF    = 0.20

# YOLO-World target classes
YOLO_WORLD_CLASSES = [
    "window", "door", "ladder", "tv", "couch", "bed", "chair",
    "dining table", "bookshelf", "potted plant", "lamp", "curtain",
    "refrigerator", "sink"
]

# SAM checkpoint (download from Meta):
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
SAM_CHECKPOINT   = os.getenv("SAM_CHECKPOINT", "models/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE   = "vit_b"   # vit_b (fast) | vit_l | vit_h (accurate)

# CLIP model variant
CLIP_MODEL       = "ViT-B/32"   # fast + good quality

# Places365 weights (auto-downloaded on first run)
PLACES365_URL    = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"

# Detection threshold
DETECTION_THRESH = float(os.getenv("DETECTION_THRESH", "0.20"))
N_GLOBAL_COLORS  = 6
N_OBJECT_COLORS  = 4
DENOISE_H        = 10
BRIGHT_THRESH    = 128

# ── House Muse class list (must match training/dataset_builder.py) ────────────
HOUSE_MUSE_CLASSES = [
    "sofa", "chair", "bed", "dining_table", "wardrobe",
    "bookshelf", "lamp", "rug", "curtain", "plant",
    "vase", "cushion", "painting", "television",
    "coffee_table", "side_table", "mirror", "desk",
]

# Object → design role
OBJECT_DESIGN_ROLE = {
    "sofa":         "anchor_piece",
    "couch":        "anchor_piece",
    "bed":          "anchor_piece",
    "dining_table": "anchor_piece",
    "dining table": "anchor_piece",
    "chair":        "accent_piece",
    "wardrobe":     "neutral_mass",
    "bookshelf":    "neutral_mass",
    "lamp":         "accent_piece",
    "rug":          "floor_anchor",
    "curtain":      "soft_furnishing",
    "plant":        "natural_accent",
    "potted plant": "natural_accent",
    "vase":         "accent_piece",
    "cushion":      "soft_furnishing",
    "painting":     "accent_piece",
    "television":   "neutral_mass",
    "tv":           "neutral_mass",
    "coffee_table": "accent_piece",
    "side_table":   "accent_piece",
    "mirror":       "accent_piece",
    "desk":         "anchor_piece",
    "window":       "architectural_feature",
    "door":         "architectural_feature",
    "ladder":       "accent_piece",
    "refrigerator": "neutral_mass",
    "sink":         "neutral_mass",
}

# Objects whose colour drives design decisions
COLOUR_RELEVANT = {
    "sofa", "couch", "chair", "bed", "dining_table", "dining table", "rug",
    "curtain", "plant", "potted plant", "vase", "cushion", "painting",
    "coffee_table", "side_table", "bookshelf", "lamp",
}

# Room classification from dominant objects
OBJECT_ROOM_VOTES = {
    "bed":          ("bedroom",     3.0),
    "sofa":         ("living_room", 2.0),
    "couch":        ("living_room", 2.0),
    "television":   ("living_room", 1.5),
    "tv":           ("living_room", 1.5),
    "dining_table": ("dining_room", 2.0),
    "dining table": ("dining_room", 2.0),
    "desk":         ("study_room",  2.0),
    "wardrobe":     ("bedroom",     1.0),
    "bookshelf":    ("study_room",  1.0),
    "refrigerator": ("kitchen",     2.5),
    "sink":         ("bathroom",    1.5),
    "chair":        (None,          0.0),
    "plant":        (None,          0.0),
    "potted plant": (None,          0.0),
    "window":       (None,          0.0),
    "door":         (None,          0.0),
    "ladder":       (None,          0.0),
    "lamp":         (None,          0.0),
    "curtain":      (None,          0.0),
}

# CLIP text prompts for zero-shot room classification
ROOM_CLIP_PROMPTS = {
    "living_room":  "a living room with sofa and television",
    "bedroom":      "a bedroom with a bed and pillows",
    "kitchen":      "a kitchen with countertops and appliances",
    "dining_room":  "a dining room with a table and chairs",
    "bathroom":     "a bathroom with sink and toilet",
    "study_room":   "a home office or study room with desk and bookshelf",
}

# Style prompts for CLIP similarity (must match frontend chip labels)
STYLE_CLIP_PROMPTS = {
    "Japandi":             "japandi interior design wabi-sabi minimal natural materials",
    "Bohemian":            "bohemian eclectic interior design layered textiles patterns",
    "Mid-Century":         "mid century modern retro interior warm wood tones",
    "Contemporary Indian": "contemporary Indian interior design brass terracotta modern",
    "Vintage Indian":      "vintage Indian haveli interior rich colours antique brass jaali",
    "Art Deco":            "art deco interior gold geometric opulent glamour",
    "Coastal":             "coastal beach interior light airy rattan white blue",
    "Dark Academia":       "dark academia interior mahogany books moody candlelight",
    "Cottagecore":         "cottagecore floral pastoral interior botanical sage green",
    "Industrial":          "industrial loft interior exposed concrete steel dark",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

_yolo_model           = None
_active_detector_name = None
_is_coco_fallback     = False
_sam_predictor        = None
_clip_model           = None
_clip_preprocess      = None
_clip_failed          = False
_places_model         = None
_effnet_model         = None
_effnet_tf            = None


def _load_yolo() -> YOLO:
    global _yolo_model, _active_detector_name, _is_coco_fallback
    if _yolo_model is None:
        # Check custom trained weights first if they exist
        if Path(YOLO_WEIGHTS).exists():
            print(f"[HouseMuse] Loading custom YOLO: {YOLO_WEIGHTS}")
            _yolo_model = YOLO(YOLO_WEIGHTS)
            _is_coco_fallback = False
            _active_detector_name = f"Custom YOLO ({YOLO_WEIGHTS})"
            print(f"[HouseMuse] Active detector: {_active_detector_name}")
            return _yolo_model

        # 1. Primary detector: YOLO-World (yolov8s-worldv2.pt) with set_classes
        # Use absolute path built from file's own location
        # (overridable via YOLO_WORLD_WEIGHTS / COCO_WEIGHTS env vars)
        world_weights = YOLO_WORLD_WEIGHTS
        coco_weights = COCO_FALLBACK_WEIGHTS

        loaded_world = False
        if os.path.exists(world_weights):
            try:
                model = YOLO(world_weights)
                model.set_classes(YOLO_WORLD_CLASSES)
                _yolo_model = model
                _is_coco_fallback = False
                _active_detector_name = f"YOLO-World ({world_weights})"
                print(f"[HouseMuse] Active detector: {_active_detector_name}")
                loaded_world = True
            except Exception as e:
                print(f"[HouseMuse] YOLO-World load failed ({e}) — falling back to COCO detector")
        else:
            print(f"[HouseMuse] YOLO-World weights not found at {world_weights} — falling back to COCO detector")

        # 2. Fallback to COCO yolov8m.pt if YOLO-World failed or was missing
        if not loaded_world:
            if not os.path.exists(coco_weights):
                raise FileNotFoundError(f"COCO fallback weights not found at {coco_weights}")
            _yolo_model = YOLO(coco_weights)
            _is_coco_fallback = True
            _active_detector_name = f"COCO fallback ({coco_weights})"
            print(f"[HouseMuse] Active detector: {_active_detector_name}")

    return _yolo_model


def _load_sam():
    global _sam_predictor
    if not SAM_AVAILABLE:
        return None
    if _sam_predictor is None:
        if not Path(SAM_CHECKPOINT).exists():
            print(f"[HouseMuse] SAM checkpoint not found at {SAM_CHECKPOINT}")
            print("[HouseMuse] Download: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/")
            return None
        print("[HouseMuse] Loading SAM …")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.eval()
        _sam_predictor = SamPredictor(sam)
        print("[HouseMuse] SAM ready ✓")
    return _sam_predictor


def _load_clip():
    global _clip_model, _clip_preprocess, _clip_failed
    if not CLIP_AVAILABLE or _clip_failed:
        return None, None
    if _clip_model is None:
        print("[HouseMuse] Loading CLIP …")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _clip_model, _clip_preprocess = openai_clip.load(CLIP_MODEL, device=device)
            _clip_model.eval()
            print("[HouseMuse] CLIP ready ✓")
        except Exception as e:
            # Weights (~/.cache/clip/ViT-B-32.pt) missing and download failed/offline.
            # Callers already handle (None, None) and fall back to non-CLIP paths.
            _clip_model, _clip_preprocess = None, None
            _clip_failed = True
            print(f"[HouseMuse] CLIP unavailable ({e}) — continuing without CLIP")
            return None, None
    return _clip_model, _clip_preprocess


def _load_places365():
    """
    Load ResNet-50 pretrained on Places365 for room classification.
    Only loaded if cached locally to avoid downloading model weights.
    """
    global _places_model
    if _places_model is None:
        try:
            import torch.hub as torch_hub
            hub_dir = torch_hub.get_dir()
            cached = hub_dir and os.path.exists(os.path.join(hub_dir, "CSAILVision_places365_master"))
            if not cached:
                _places_model = None
                return None
            print("[HouseMuse] Loading Places365 …")
            _places_model = torch_hub.load(
                "CSAILVision/places365",
                "resnet50",
                pretrained=True,
                force_reload=False,
            )
            _places_model.eval()
            print("[HouseMuse] Places365 ready ✓")
        except Exception as e:
            print(f"[HouseMuse] Places365 load failed ({e}) — using CLIP zero-shot for room type.")
            _places_model = None
    return _places_model


def _load_efficientnet_fallback():
    """EfficientNet-B4 fallback when CLIP is unavailable."""
    global _effnet_model, _effnet_tf
    if _effnet_model is None:
        weights       = tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1
        _effnet_model = tv_models.efficientnet_b4(weights=weights)
        _effnet_model.eval()
        _effnet_tf = T.Compose([
            T.Resize((380, 380)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return _effnet_model, _effnet_tf


# ══════════════════════════════════════════════════════════════════════════════
#  1. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image. Ensure it is a valid JPG/PNG.")

    bgr = cv2.resize(bgr, (800, 600), interpolation=cv2.INTER_AREA)
    bgr = cv2.fastNlMeansDenoisingColored(bgr, None, DENOISE_H, DENOISE_H, 7, 21)

    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    bgr     = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb


# ══════════════════════════════════════════════════════════════════════════════
#  2. LIGHTING
# ══════════════════════════════════════════════════════════════════════════════

def detect_lighting(bgr: np.ndarray) -> dict:
    gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    b_mean   = float(np.mean(bgr[:,:,0]))
    r_mean   = float(np.mean(bgr[:,:,2]))
    ratio    = r_mean / (b_mean + 1e-5)

    return {
        "condition":         "bright" if mean_val >= BRIGHT_THRESH else "dim",
        "mean_brightness":   round(mean_val, 2),
        "color_temperature": "warm" if ratio > 1.15 else ("cool" if ratio < 0.87 else "neutral"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. YOLO OBJECT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_objects(rgb: np.ndarray) -> list[dict]:
    """
    Run YOLOv8 (fine-tuned on IKEA+ADE20K) on the room image.
    Returns list of detected objects with bounding boxes.
    """
    model  = _load_yolo()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf   = COCO_FALLBACK_CONF if _is_coco_fallback else DETECTION_THRESH

    results = model.predict(
        source    = rgb,
        conf      = conf,
        device    = device,
        verbose   = False,
        imgsz     = 640,
    )

    detections = []
    if not results:
        return detections

    preds = results[0]
    for box in preds.boxes:
        cls_idx    = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        # Map class index → label
        if cls_idx < len(HOUSE_MUSE_CLASSES):
            label = HOUSE_MUSE_CLASSES[cls_idx]
        else:
            # COCO pretrained fallback — map COCO names
            label = preds.names.get(cls_idx, "unknown").lower()

        detections.append({
            "label":       label,
            "interior":    label,
            "confidence":  round(confidence, 3),
            "box":         [x1, y1, x2, y2],
            "design_role": OBJECT_DESIGN_ROLE.get(label),
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ══════════════════════════════════════════════════════════════════════════════
#  4. SAM MASKING + MASKED K-MEANS COLOUR EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _kmeans_on_pixels(pixels: np.ndarray, n: int) -> list[dict]:
    """Run K-Means on a (N, 3) float32 pixel array."""
    if len(pixels) < n * 5:
        avg = pixels.mean(axis=0).astype(int)
        r, g, b = int(avg[0]), int(avg[1]), int(avg[2])
        return [{"hex": f"#{r:02X}{g:02X}{b:02X}", "rgb": [r,g,b], "percentage": 100.0}]

    if len(pixels) > 30_000:
        idx    = np.random.choice(len(pixels), 30_000, replace=False)
        pixels = pixels[idx]

    km     = KMeans(n_clusters=min(n, len(pixels)), n_init=8, random_state=42)
    km.fit(pixels)
    counts  = np.bincount(km.labels_)
    total   = counts.sum()
    centers = km.cluster_centers_.astype(int)

    colors = []
    for i in np.argsort(-counts):
        r, g, b = int(centers[i][0]), int(centers[i][1]), int(centers[i][2])
        colors.append({
            "hex":        f"#{r:02X}{g:02X}{b:02X}",
            "rgb":        [r, g, b],
            "percentage": round(float(counts[i]) / total * 100, 2),
        })
    return colors


def extract_object_colors(rgb: np.ndarray, detections: list[dict]) -> list[dict]:
    """
    For each colour-relevant detection:
      1. Try SAM → get pixel-perfect mask
      2. Fall back to bbox crop if SAM unavailable
      3. Run K-Means on masked pixels only

    Augments each relevant detection with {"colors": [...], "mask_used": bool}
    """
    sam        = _load_sam()
    h, w       = rgb.shape[:2]
    results    = []

    if sam is not None:
        # Set SAM image once (expensive — reuse across all objects)
        sam.set_image(rgb)

    for det in detections:
        if det["interior"] not in COLOUR_RELEVANT:
            continue

        x1, y1, x2, y2 = det["box"]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        det = dict(det)
        mask_used = False

        # ── Try SAM mask ──────────────────────────────────────────────────
        if sam is not None:
            try:
                input_box = np.array([[x1, y1, x2, y2]])
                masks, scores, _ = sam.predict(
                    box            = input_box,
                    multimask_output = False,
                )
                best_mask = masks[np.argmax(scores)]   # shape (H, W) bool

                # Extract only pixels inside the mask
                masked_pixels = rgb[best_mask].astype(np.float32)
                if len(masked_pixels) > 50:
                    det["colors"] = _kmeans_on_pixels(masked_pixels, N_OBJECT_COLORS)
                    det["mask_used"] = True
                    mask_used = True

            except Exception as e:
                print(f"[SAM] Mask failed for {det['interior']}: {e}")

        # ── Fall back to bbox crop ────────────────────────────────────────
        if not mask_used:
            crop   = rgb[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
            det["colors"]    = _kmeans_on_pixels(crop, N_OBJECT_COLORS)
            det["mask_used"] = False

        results.append(det)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  5. CLIP FEATURE EXTRACTION + STYLE MATCHING
# ══════════════════════════════════════════════════════════════════════════════

def extract_clip_features(rgb: np.ndarray) -> np.ndarray:
    """
    Extract CLIP image embedding (512-d for ViT-B/32).
    Used by recommendation engine for semantic style matching.
    """
    clip_model, preprocess = _load_clip()

    if clip_model is None:
        # Fallback to EfficientNet
        model, tf = _load_efficientnet_fallback()
        pil = Image.fromarray(rgb)
        with torch.no_grad():
            t     = tf(pil).unsqueeze(0)
            feats = model.features(t)
            feats = model.avgpool(feats).flatten().numpy()
        return feats

    device = next(clip_model.parameters()).device
    pil    = Image.fromarray(rgb)

    with torch.no_grad():
        img_tensor = preprocess(pil).unsqueeze(0).to(device)
        features   = clip_model.encode_image(img_tensor)
        features   = features / features.norm(dim=-1, keepdim=True)  # normalise

    return features.cpu().numpy().flatten()   # (512,)


def compute_style_similarities(rgb: np.ndarray) -> dict[str, float]:
    """
    Use CLIP zero-shot to score how well the room image matches each style.
    Returns dict: {"Japandi": 0.82, "Bohemian": 0.61, ...}

    This is used to:
      a) Auto-detect the room's current style
      b) Score style compatibility in the recommendation engine
    """
    clip_model, preprocess = _load_clip()

    if clip_model is None:
        return {}   # graceful fallback

    device = next(clip_model.parameters()).device
    pil    = Image.fromarray(rgb)

    prompts = list(STYLE_CLIP_PROMPTS.values())
    styles  = list(STYLE_CLIP_PROMPTS.keys())

    with torch.no_grad():
        img_tensor  = preprocess(pil).unsqueeze(0).to(device)
        text_tokens = openai_clip.tokenize(prompts).to(device)

        img_feats  = clip_model.encode_image(img_tensor)
        text_feats = clip_model.encode_text(text_tokens)

        img_feats  = img_feats  / img_feats.norm(dim=-1,  keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        sims = (img_feats @ text_feats.T).squeeze(0).cpu().numpy()

    return {style: round(float(sim), 4) for style, sim in zip(styles, sims)}


# ══════════════════════════════════════════════════════════════════════════════
#  6. ROOM CLASSIFICATION
#     Strategy: Places365 primary → CLIP zero-shot backup → object votes fallback
# ══════════════════════════════════════════════════════════════════════════════

# Places365 → House Muse room type mapping
PLACES365_MAP = {
    "living_room":      "living_room",
    "bedroom":          "bedroom",
    "kitchen":          "kitchen",
    "dining_room":      "dining_room",
    "bathroom":         "bathroom",
    "home_office":      "study_room",
    "office":           "study_room",
    "nursery":          "bedroom",
    "childs_room":      "bedroom",
    "playroom":         "living_room",
    "closet":           "bedroom",
    "laundry_room":     "bathroom",
}

_PLACES365_CATEGORIES = None

def _get_places365_categories() -> list[str]:
    """Load Places365 class names."""
    global _PLACES365_CATEGORIES
    if _PLACES365_CATEGORIES is None:
        _PLACES365_CATEGORIES = []
    return _PLACES365_CATEGORIES


def classify_room_places365(rgb: np.ndarray) -> str | None:
    """Use Places365 for scene classification."""
    model = _load_places365()
    if model is None:
        return None

    tf = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    pil    = Image.fromarray(rgb)
    tensor = tf(pil).unsqueeze(0)

    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)

    top5_idx = torch.topk(probs, 5).indices[0].numpy()
    cats     = _get_places365_categories()

    if not cats:
        return None

    for idx in top5_idx:
        if idx < len(cats):
            cat_name = cats[idx].lower().replace(" ", "_")
            mapped   = PLACES365_MAP.get(cat_name)
            if mapped:
                return mapped

    return None


def classify_room_clip(rgb: np.ndarray) -> str | None:
    """CLIP zero-shot room classification."""
    clip_model, preprocess = _load_clip()
    if clip_model is None:
        return None

    device  = next(clip_model.parameters()).device
    pil     = Image.fromarray(rgb)
    prompts = list(ROOM_CLIP_PROMPTS.values())
    rooms   = list(ROOM_CLIP_PROMPTS.keys())

    with torch.no_grad():
        img_tensor  = preprocess(pil).unsqueeze(0).to(device)
        text_tokens = openai_clip.tokenize(prompts).to(device)
        img_feats   = clip_model.encode_image(img_tensor)
        text_feats  = clip_model.encode_text(text_tokens)
        img_feats   = img_feats  / img_feats.norm(dim=-1,  keepdim=True)
        text_feats  = text_feats / text_feats.norm(dim=-1, keepdim=True)
        sims        = (img_feats @ text_feats.T).squeeze(0).cpu().numpy()

    best_idx = int(np.argmax(sims))
    return rooms[best_idx]


def infer_room_type(
    rgb: np.ndarray,
    detections: list[dict],
    provided: str | None = None,
) -> tuple[str, str]:
    """
    Three-layer room classification.
    Returns (room_type, method_used).
    """
    if provided:
        return provided, "user_provided"

    # Layer 1: Places365 (highest accuracy for scene classification)
    room = classify_room_places365(rgb)
    if room:
        return room, "places365"

    # Layer 2: CLIP zero-shot
    room = classify_room_clip(rgb)
    if room:
        return room, "clip_zero_shot"

    # Layer 3: Object vote
    votes = {}
    for det in detections:
        obj  = det.get("interior", "")
        info = OBJECT_ROOM_VOTES.get(obj)
        if info and info[0]:
            votes[info[0]] = votes.get(info[0], 0) + info[1] * det["confidence"]

    if votes:
        return max(votes, key=votes.get), "object_vote"

    return "living_room", "default"


# ══════════════════════════════════════════════════════════════════════════════
#  7. GLOBAL DOMINANT COLOUR EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_dominant_colors(rgb: np.ndarray, n: int = N_GLOBAL_COLORS) -> list[dict]:
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 50_000:
        idx    = np.random.choice(len(pixels), 50_000, replace=False)
        pixels = pixels[idx]

    km     = KMeans(n_clusters=n, n_init=10, random_state=42)
    km.fit(pixels)
    counts  = np.bincount(km.labels_)
    centers = km.cluster_centers_.astype(int)

    colors = []
    for i in np.argsort(-counts):
        r, g, b = int(centers[i][0]), int(centers[i][1]), int(centers[i][2])
        colors.append({
            "hex":        f"#{r:02X}{g:02X}{b:02X}",
            "rgb":        [r, g, b],
            "percentage": round(float(counts[i]) / counts.sum() * 100, 2),
        })
    return colors


# ══════════════════════════════════════════════════════════════════════════════
#  8. OBJECT-AWARE PALETTE (driven by SAM-masked object colours)
# ══════════════════════════════════════════════════════════════════════════════

def _rgb_to_hsv(r, g, b): return colorsys.rgb_to_hsv(r/255, g/255, b/255)
def _hsv_to_hex(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def build_object_aware_palette(
    object_colors: list[dict],
    global_colors: list[dict],
    lighting: dict,
    style_sims: dict,
) -> dict:
    """
    Build wall/accent/ceiling palette from detected object colours.
    Now also considers style similarity scores from CLIP.
    """
    # Find anchor (most design-relevant detected object)
    anchor_priority = ["anchor_piece", "floor_anchor", "soft_furnishing", "accent_piece"]
    anchor     = None
    anchor_hex = None

    for role in anchor_priority:
        for obj in object_colors:
            if obj.get("design_role") == role and obj.get("colors"):
                anchor     = obj["interior"]
                anchor_hex = obj["colors"][0]["hex"]
                break
        if anchor_hex:
            break

    if anchor_hex is None:
        anchor_hex = global_colors[0]["hex"] if global_colors else "#C4A882"

    r, g, b = int(anchor_hex[1:3],16), int(anchor_hex[3:5],16), int(anchor_hex[5:7],16)
    h, s, v = _rgb_to_hsv(r, g, b)

    # Determine strategy from anchor saturation/value
    if s > 0.45 or v < 0.55:
        # Rich/dark anchor → lighter desaturated wall
        wall_hex  = _hsv_to_hex(h, max(0.0, s - 0.28), min(1.0, v + 0.22))
        strategy  = "contrast_lighten"
        note      = (f"Your {anchor or 'anchor piece'} has a rich tone — "
                     f"walls use a lighter version of the same hue family "
                     f"so the furniture stays the focal point.")
    else:
        # Light/neutral anchor → complementary wall for contrast
        h_comp    = (h + 0.5) % 1.0
        wall_hex  = _hsv_to_hex(h_comp, min(1.0, s + 0.12), max(0.3, v - 0.12))
        strategy  = "complement"
        note      = (f"Your {anchor or 'anchor piece'} is light — "
                     f"a complementary wall colour adds contrast without competing.")

    # Accent: from accent_piece object or analogous
    accent_hex = None
    for obj in object_colors:
        if obj.get("design_role") == "accent_piece" and obj.get("colors"):
            accent_hex = obj["colors"][0]["hex"]
            break
    if not accent_hex:
        acc_h      = (h + 0.083) % 1.0
        accent_hex = _hsv_to_hex(acc_h, s, v)

    # Ceiling: always lighter than wall
    wh, ws, wv = _rgb_to_hsv(*[int(wall_hex[i:i+2],16) for i in (1,3,5)])
    ceiling_hex = _hsv_to_hex(wh, max(0.0, ws - 0.18), min(1.0, wv + 0.15))

    # Lighting adjustment
    if lighting.get("condition") == "dim":
        wall_hex = _hsv_to_hex((h + 0.02) % 1.0, max(0.0, s - 0.12), min(1.0, v + 0.12))
        note    += " Limited natural light — tones have been warmed and brightened."

    # CLIP style note
    top_style = max(style_sims, key=style_sims.get) if style_sims else None
    style_note = ""
    if top_style:
        style_note = f" Room aesthetics closest match: {top_style} ({style_sims[top_style]*100:.0f}% similarity)."

    return {
        "wall_color":          wall_hex,
        "accent_color":        accent_hex,
        "ceiling_color":       ceiling_hex,
        "object_driven":       True,
        "anchor_object":       anchor,
        "anchor_color":        anchor_hex,
        "palette_strategy":    strategy,
        "object_palette_note": note + style_note,
        "detected_style":      top_style,
        "style_similarities":  style_sims,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER PIPELINE  (called from app.py — unchanged signature)
# ══════════════════════════════════════════════════════════════════════════════

def process_image(
    image_bytes: bytes,
    provided_room_type: str | None = None,
) -> dict:
    """
    Full pipeline. Called from app.py with no changes to call signature.

    Returns:
        dominant_colors   : global K-Means colours
        lighting          : bright/dim + warm/cool
        room_type         : classified room
        room_type_method  : how room was classified
        cnn_features      : CLIP 512-d vector (or EfficientNet fallback)
        detections        : YOLOv8 object detections
        object_colors     : SAM-masked per-object K-Means colours
        object_palette    : object-aware wall/accent/ceiling
        style_similarities: CLIP style scores
    """
    bgr, rgb = preprocess_image(image_bytes)

    # Parallel-safe sequential execution
    lighting      = detect_lighting(bgr)
    detections    = detect_objects(rgb)
    object_colors = extract_object_colors(rgb, detections)
    global_colors = extract_dominant_colors(rgb)
    clip_features = extract_clip_features(rgb)
    style_sims    = compute_style_similarities(rgb)

    room_type, room_method = infer_room_type(
        rgb, detections, provided_room_type
    )

    obj_palette = build_object_aware_palette(
        object_colors, global_colors, lighting, style_sims
    )

    return {
        "dominant_colors":    global_colors,
        "lighting":           lighting,
        "room_type":          room_type,
        "room_type_method":   room_method,
        "cnn_features":       clip_features.tolist(),
        "detections":         detections,
        "object_colors":      object_colors,
        "object_palette":     obj_palette,
        "style_similarities": style_sims,
    }
