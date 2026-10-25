"""
House Muse — Model Evaluation Suite
════════════════════════════════════════════════════════════════════════════════
Autonomous evaluation of House-Muse ML pipeline models:
- Object Detectors: COCO YOLOv8m vs YOLO-World v2
- Structure Preservation: Canny-Edge SSIM and Depth Pearson Correlation (512px)
- Style Alignment: CLIP ViT-B/32 Zero-Shot Similarity, Top-1 Accuracy, Confusion Matrix
- Exports: PNG figures (300 dpi), results.json, results.csv, evaluation_figures.zip
════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import shutil
import json
import zipfile
from pathlib import Path

# Print model download sizes before downloading/loading
print("=" * 70)
print("HOUSE MUSE MODEL EVALUATION")
print("=" * 70)
print("Hugging Face model download sizes before downloading:")
print("  - LiheYoung/depth-anything-small-hf (Depth Estimator): 94.58 MB")
print("  - openai/clip-vit-base-patch32 (CLIP Vision-Language): 580.66 MB")
print("  - Total model download size: 675.24 MB")
print("=" * 70)

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import scipy.stats
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, CLIPProcessor, CLIPModel
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# PATHS CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # house-muse/ml
EVAL_DIR = BASE_DIR / "evaluation"
INPUTS_DIR = EVAL_DIR / "inputs"
FIGURES_DIR = EVAL_DIR / "figures"

INPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

WORKSPACE_ROOT = BASE_DIR.parent  # house-muse

# Detectors
COCO_WEIGHTS = BASE_DIR / "yolov8m.pt"
WORLD_WEIGHTS = EVAL_DIR / "yolov8s-worldv2.pt"
if not WORLD_WEIGHTS.exists() and (BASE_DIR / "yolov8s-worldv2.pt").exists():
    WORLD_WEIGHTS = BASE_DIR / "yolov8s-worldv2.pt"

# Define 9 Styles Mapping
# 3 from ml/test_outputs/ (input = frontend/public/test/test.jpg)
# 6 from backend/media/projects/generated/ (input = backend/media/projects/source/...)
STYLE_DATA = [
    {
        "style": "contemporary_indian",
        "display_name": "Contemporary Indian",
        "output_source": BASE_DIR / "test_outputs" / "test_output_contemporary_indian.jpg",
        "input_source": WORKSPACE_ROOT / "frontend" / "public" / "test" / "test.jpg",
        "origin": "ml/test_outputs",
    },
    {
        "style": "industrial",
        "display_name": "Industrial",
        "output_source": BASE_DIR / "test_outputs" / "test_output_industrial.jpg",
        "input_source": WORKSPACE_ROOT / "frontend" / "public" / "test" / "test.jpg",
        "origin": "ml/test_outputs",
    },
    {
        "style": "mid_century",
        "display_name": "Mid-Century Modern",
        "output_source": BASE_DIR / "test_outputs" / "test_output_mid_century.jpg",
        "input_source": WORKSPACE_ROOT / "frontend" / "public" / "test" / "test.jpg",
        "origin": "ml/test_outputs",
    },
    {
        "style": "victorian",
        "display_name": "Victorian",
        "output_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "generated" / "gen_2_1791657524.jpg",
        "input_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "source" / "room.jpg",
        "origin": "backend/projects/gen_2",
    },
    {
        "style": "scandinavian",
        "display_name": "Scandinavian",
        "output_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "generated" / "gen_3_1791658544.jpg",
        "input_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "source" / "room1.jpg",
        "origin": "backend/projects/gen_3",
    },
    {
        "style": "vintage_indian",
        "display_name": "Vintage Indian",
        "output_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "generated" / "gen_4_1791658670.jpg",
        "input_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "source" / "room_2.webp",
        "origin": "backend/projects/gen_4",
    },
    {
        "style": "shabby_chic_style",
        "display_name": "Shabby Chic",
        "output_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "generated" / "gen_5_1791658779.jpg",
        "input_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "source" / "room_2_qCfddJR.webp",
        "origin": "backend/projects/gen_5",
    },
    {
        "style": "eclectic",
        "display_name": "Eclectic",
        "output_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "generated" / "gen_6_1791658974.jpg",
        "input_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "source" / "room_2_RhLDUMC.webp",
        "origin": "backend/projects/gen_6",
    },
    {
        "style": "coastal",
        "display_name": "Coastal",
        "output_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "generated" / "gen_14_1792607326.jpg",
        "input_source": WORKSPACE_ROOT / "backend" / "media" / "projects" / "source" / "room_2_1pt4VK0.webp",
        "origin": "backend/projects/gen_14",
    },
]

# Copy images into ml/evaluation/inputs/ without modifying originals
print("\n[Step 1] Copying evaluation images to ml/evaluation/inputs/...")
for item in STYLE_DATA:
    style = item["style"]
    src_out = Path(item["output_source"])
    src_in = Path(item["input_source"])

    dst_out = INPUTS_DIR / f"output_{style}{src_out.suffix}"
    dst_in = INPUTS_DIR / f"input_{style}{src_in.suffix}"

    if not dst_out.exists() or dst_out.stat().st_size != src_out.stat().st_size:
        shutil.copy2(src_out, dst_out)
    if not dst_in.exists() or dst_in.stat().st_size != src_in.stat().st_size:
        shutil.copy2(src_in, dst_in)

    item["local_output"] = dst_out
    item["local_input"] = dst_in
    print(f"  [{style}]")
    print(f"    Output: {src_out.name} -> {dst_out.name} ({dst_out.stat().st_size} bytes)")
    print(f"    Input:  {src_in.name} -> {dst_in.name} ({dst_in.stat().st_size} bytes)")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2] Loading Models...")

# YOLOv8m (COCO)
print(f"Loading YOLOv8m (COCO): {COCO_WEIGHTS}")
yolo_coco = YOLO(str(COCO_WEIGHTS))

# YOLOv8s-worldv2
print(f"Loading YOLOv8s-worldv2: {WORLD_WEIGHTS}")
yolo_world = YOLO(str(WORLD_WEIGHTS))

# Depth Model (LiheYoung/depth-anything-small-hf)
print("Loading Depth Estimator (LiheYoung/depth-anything-small-hf)...")
depth_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
depth_model.eval()

# CLIP Model (openai/clip-vit-base-patch32)
print("Loading CLIP (openai/clip-vit-base-patch32)...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Configure YOLO-World target classes with interior vocabulary
INTERIOR_CLASSES = [
    "window", "door", "ladder", "tv", "couch", "bed", "chair",
    "dining table", "bookshelf", "potted plant", "lamp", "curtain", "sink"
]
print("Configuring YOLO-World text embeddings for interior classes via HF CLIP...")
clip_tok = clip_processor.tokenizer
tok_inputs = clip_tok(INTERIOR_CLASSES, padding=True, return_tensors="pt")
with torch.no_grad():
    text_feat = clip_model.get_text_features(**tok_inputs)
    feat = text_feat[0] if isinstance(text_feat, (tuple, list)) else (text_feat.pooler_output if hasattr(text_feat, "pooler_output") else text_feat)
    feat = feat / feat.norm(dim=-1, keepdim=True)

yolo_world.model.txt_feats = feat.unsqueeze(0)
yolo_world.model.model[-1].nc = len(INTERIOR_CLASSES)
yolo_world.model.names = {i: c for i, c in enumerate(INTERIOR_CLASSES)}
print("Models loaded successfully!\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. RUN METRIC EVALUATIONS
# ─────────────────────────────────────────────────────────────────────────────
results_per_style = []
coco_detections_all = {"inputs": [], "outputs": []}
world_detections_all = {"inputs": [], "outputs": []}

# Setup style prompts for CLIP
styles_list = [item["style"] for item in STYLE_DATA]
style_prompts = [f"a {item['display_name'].lower()} style interior room" for item in STYLE_DATA]
prompt_inputs = clip_processor(text=style_prompts, return_tensors="pt", padding=True)
with torch.no_grad():
    prompt_feats = clip_model.get_text_features(**prompt_inputs)
    if hasattr(prompt_feats, "pooler_output"):
        prompt_feats = prompt_feats.pooler_output
    prompt_feats = prompt_feats / prompt_feats.norm(dim=-1, keepdim=True)

print("[Step 3] Evaluating 9 styles...")

# Pre-encode images and run evaluations
image_clip_embeddings = []

for idx, item in enumerate(STYLE_DATA):
    style = item["style"]
    display_name = item["display_name"]
    inp_path = item["local_input"]
    out_path = item["local_output"]

    # Open images
    im_in_raw = Image.open(inp_path).convert("RGB")
    im_out_raw = Image.open(out_path).convert("RGB")

    # Resize to 512x512 for structural comparison
    im_in_512 = im_in_raw.resize((512, 512), Image.Resampling.BILINEAR)
    im_out_512 = im_out_raw.resize((512, 512), Image.Resampling.BILINEAR)

    # 1. Canny-Edge SSIM
    g_in = cv2.cvtColor(np.array(im_in_512), cv2.COLOR_RGB2GRAY)
    g_out = cv2.cvtColor(np.array(im_out_512), cv2.COLOR_RGB2GRAY)
    e_in = cv2.Canny(g_in, 100, 200)
    e_out = cv2.Canny(g_out, 100, 200)
    canny_ssim_val = float(ssim(e_in, e_out))

    # 2. Depth Pearson Correlation
    d_inputs_in = depth_processor(images=im_in_512, return_tensors="pt")
    d_inputs_out = depth_processor(images=im_out_512, return_tensors="pt")
    with torch.no_grad():
        d_map_in = depth_model(**d_inputs_in).predicted_depth.squeeze().cpu().numpy()
        d_map_out = depth_model(**d_inputs_out).predicted_depth.squeeze().cpu().numpy()
    d_map_in_res = cv2.resize(d_map_in, (512, 512))
    d_map_out_res = cv2.resize(d_map_out, (512, 512))
    depth_pearson_corr, _ = scipy.stats.pearsonr(d_map_in_res.flatten(), d_map_out_res.flatten())
    depth_pearson_corr = float(depth_pearson_corr)

    # 3. CLIP Style Alignment
    c_inputs = clip_processor(images=im_out_raw, return_tensors="pt")
    with torch.no_grad():
        im_feat = clip_model.get_image_features(**c_inputs)
        if hasattr(im_feat, "pooler_output"):
            im_feat = im_feat.pooler_output
        im_feat = im_feat / im_feat.norm(dim=-1, keepdim=True)
    image_clip_embeddings.append(im_feat)

    sims = (im_feat @ prompt_feats.T).squeeze().cpu().numpy()
    target_clip_sim = float(sims[idx])
    pred_idx = int(sims.argmax())
    pred_style = styles_list[pred_idx]
    is_top1_match = bool(pred_idx == idx)

    # 4. Object Detection on Input and Output
    # Run YOLOv8m (COCO)
    coco_res_in = yolo_coco(str(inp_path), verbose=False)[0]
    coco_res_out = yolo_coco(str(out_path), verbose=False)[0]

    # Run YOLO-World
    world_res_in = yolo_world(str(inp_path), verbose=False)[0]
    world_res_out = yolo_world(str(out_path), verbose=False)[0]

    def extract_detections(res, model):
        dets = []
        for b in res.boxes:
            c_id = int(b.cls[0])
            conf = float(b.conf[0])
            name = model.names[c_id]
            dets.append({"class": name, "confidence": conf})
        return dets

    c_dets_in = extract_detections(coco_res_in, yolo_coco)
    c_dets_out = extract_detections(coco_res_out, yolo_coco)
    w_dets_in = extract_detections(world_res_in, yolo_world)
    w_dets_out = extract_detections(world_res_out, yolo_world)

    coco_detections_all["inputs"].extend(c_dets_in)
    coco_detections_all["outputs"].extend(c_dets_out)
    world_detections_all["inputs"].extend(w_dets_in)
    world_detections_all["outputs"].extend(w_dets_out)

    style_record = {
        "style": style,
        "display_name": display_name,
        "origin": item["origin"],
        "input_source": str(item["input_source"].name),
        "output_source": str(item["output_source"].name),
        "canny_edge_ssim": round(canny_ssim_val, 4),
        "depth_pearson_corr": round(depth_pearson_corr, 4),
        "clip_target_similarity": round(target_clip_sim, 4),
        "clip_predicted_style": pred_style,
        "clip_top1_match": is_top1_match,
        "clip_all_similarities": {s: round(float(v), 4) for s, v in zip(styles_list, sims)},
        "coco_detections_in_count": len(c_dets_in),
        "coco_detections_out_count": len(c_dets_out),
        "world_detections_in_count": len(w_dets_in),
        "world_detections_out_count": len(w_dets_out),
    }
    results_per_style.append(style_record)
    print(f"  Done {display_name:20s} | SSIM: {canny_ssim_val:.4f} | Depth Corr: {depth_pearson_corr:.4f} | CLIP Sim: {target_clip_sim:.4f} | Top-1: {is_top1_match}")

# Compute 9x9 Confusion Matrix & Similarity Matrix
all_im_feats = torch.cat(image_clip_embeddings, dim=0)
similarity_matrix_9x9 = (all_im_feats @ prompt_feats.T).cpu().numpy()
confusion_matrix_9x9 = np.zeros((9, 9), dtype=int)
for i in range(9):
    pred = int(similarity_matrix_9x9[i].argmax())
    confusion_matrix_9x9[i, pred] += 1

top1_accuracy = float(np.trace(confusion_matrix_9x9) / 9.0)
mean_ssim = float(np.mean([r["canny_edge_ssim"] for r in results_per_style]))
mean_depth_corr = float(np.mean([r["depth_pearson_corr"] for r in results_per_style]))
mean_clip_sim = float(np.mean([r["clip_target_similarity"] for r in results_per_style]))

print(f"\nOverall Summary (n=9):")
print(f"  Mean Canny-Edge SSIM:        {mean_ssim:.4f}")
print(f"  Mean Depth Correlation:      {mean_depth_corr:.4f}")
print(f"  Mean CLIP Target Similarity: {mean_clip_sim:.4f}")
print(f"  CLIP Top-1 Style Accuracy:   {top1_accuracy*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DETECTION METRICS COMPILATION
# ─────────────────────────────────────────────────────────────────────────────
def compile_det_stats(detections):
    df = pd.DataFrame(detections)
    if df.empty:
        return {}
    stats = {}
    for cls_name, grp in df.groupby("class"):
        stats[cls_name] = {
            "count": int(len(grp)),
            "mean_confidence": round(float(grp["confidence"].mean()), 4)
        }
    return stats

coco_in_stats = compile_det_stats(coco_detections_all["inputs"])
coco_out_stats = compile_det_stats(coco_detections_all["outputs"])
world_in_stats = compile_det_stats(world_detections_all["inputs"])
world_out_stats = compile_det_stats(world_detections_all["outputs"])

# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPORT JSON AND CSV
# ─────────────────────────────────────────────────────────────────────────────
not_computed_info = {
    "custom_best_pt_detector": {
        "status": "not computed",
        "reason": "Weights file models/checkpoints/house_muse_yolo/weights/best.pt not found on disk"
    },
    "sam_segmentation": {
        "status": "not computed",
        "reason": "segment-anything package and sam_vit_b_01ec64.pth checkpoint not installed locally"
    },
    "blip_captioning": {
        "status": "not computed",
        "reason": "Remote Salesforce/blip-image-captioning-large API call prohibited to ensure zero GPU quota and zero external API dependencies"
    },
    "lora_training_loss": {
        "status": "not computed",
        "reason": "LoRA training logs (trainer_state.json, tensorboard events) not found in repo; ask the teammate who trained them"
    },
    "labelled_detection_ground_truth": {
        "status": "not computed",
        "reason": "No ground truth bounding box dataset found; evaluated via detection counts and confidence rather than precision/recall/mAP"
    }
}

full_results = {
    "sample_size": 9,
    "evaluated_styles": styles_list,
    "overall_metrics": {
        "mean_canny_edge_ssim": round(mean_ssim, 4),
        "mean_depth_pearson_correlation": round(mean_depth_corr, 4),
        "mean_clip_target_similarity": round(mean_clip_sim, 4),
        "clip_zero_shot_top1_accuracy": round(top1_accuracy, 4),
    },
    "style_evaluations": results_per_style,
    "detection_analysis": {
        "note": "no ground truth: detections and confidence, not accuracy",
        "yolov8m_coco": {
            "input_photos": coco_in_stats,
            "generated_outputs": coco_out_stats
        },
        "yolov8s_worldv2": {
            "input_photos": world_in_stats,
            "generated_outputs": world_out_stats
        }
    },
    "metrics_not_computed": not_computed_info
}

# Save results.json
json_path = EVAL_DIR / "results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(full_results, f, indent=2)
print(f"\nSaved results JSON to: {json_path}")

# Save results.csv
df_summary = pd.DataFrame([{
    "Style": r["style"],
    "Display Name": r["display_name"],
    "Origin": r["origin"],
    "Compared Against Input": r["input_source"],
    "Canny Edge SSIM": r["canny_edge_ssim"],
    "Depth Pearson Corr": r["depth_pearson_corr"],
    "CLIP Similarity": r["clip_target_similarity"],
    "CLIP Predicted Style": r["clip_predicted_style"],
    "Top-1 Match": r["clip_top1_match"],
    "YOLOv8m In Dets": r["coco_detections_in_count"],
    "YOLOv8m Out Dets": r["coco_detections_out_count"],
    "YOLO-World In Dets": r["world_detections_in_count"],
    "YOLO-World Out Dets": r["world_detections_out_count"],
} for r in results_per_style])
csv_path = EVAL_DIR / "results.csv"
df_summary.to_csv(csv_path, index=False)
print(f"Saved results CSV to: {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERATE PUBLICATION-GRADE FIGURES (300 DPI, WHITE BG)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 4] Generating publication-quality figures at 300 DPI...")

# Styling constants
COLOR_PRIMARY = "#1f77b4"     # Steel Blue
COLOR_SECONDARY = "#2ca02c"   # Forest Green
COLOR_ACCENT = "#ff7f0e"      # Amber
COLOR_HIGHLIGHT = "#d62728"   # Crimson
COLOR_MUTED = "#7f7f7f"       # Gray
FONT_FAMILY = "sans-serif"
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# ── FIGURE 1: Detection Metrics ──────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

highlight_classes = ["window", "door", "ladder", "tv"]
common_classes = ["tv", "couch", "chair", "bed", "dining table", "potted plant", "lamp", "window", "door", "ladder", "bookshelf", "sink"]

# YOLOv8m (COCO) counts vs YOLO-World counts on generated outputs
coco_counts = [coco_out_stats.get(c, {}).get("count", 0) for c in common_classes]
world_counts = [world_out_stats.get(c, {}).get("count", 0) for c in common_classes]

x = np.arange(len(common_classes))
width = 0.38

bars_coco = ax1.bar(x - width/2, coco_counts, width, label="YOLOv8m (COCO)", color="#4682b4", edgecolor="#2f4f4f", linewidth=0.8)
bars_world = ax1.bar(x + width/2, world_counts, width, label="YOLOv8s-World v2", color="#e07a5f", edgecolor="#8b3a2b", linewidth=0.8)

for bar in bars_coco:
    h = bar.get_height()
    if h > 0:
        ax1.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar in bars_world:
    h = bar.get_height()
    if h > 0:
        ax1.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Highlight specific target classes on x-axis
x_labels = [f"★ {c.upper()}" if c in highlight_classes else c for c in common_classes]
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
ax1.set_ylabel("Total Detection Count Across 9 Outputs", fontsize=10, fontweight="bold")
ax1.set_title("Total Detections per Class (Generated Outputs)", fontsize=11, fontweight="bold")
ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
ax1.grid(axis="y", linestyle="--", alpha=0.3)
ax1.set_ylim(0, max(max(coco_counts), max(world_counts)) + 3)

# Subplot 2: Mean Confidence Comparison
coco_confs = [coco_out_stats.get(c, {}).get("mean_confidence", 0.0) for c in common_classes]
world_confs = [world_out_stats.get(c, {}).get("mean_confidence", 0.0) for c in common_classes]

bars_c_conf = ax2.bar(x - width/2, coco_confs, width, label="YOLOv8m (COCO)", color="#52796f", edgecolor="#2f3e46", linewidth=0.8)
bars_w_conf = ax2.bar(x + width/2, world_confs, width, label="YOLOv8s-World v2", color="#81b29a", edgecolor="#354f52", linewidth=0.8)

for bar in bars_c_conf:
    h = bar.get_height()
    if h > 0:
        ax2.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)
for bar in bars_w_conf:
    h = bar.get_height()
    if h > 0:
        ax2.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)

ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
ax2.set_ylabel("Mean Detection Confidence", fontsize=10, fontweight="bold")
ax2.set_title("Mean Detection Confidence per Class", fontsize=11, fontweight="bold")
ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
ax2.grid(axis="y", linestyle="--", alpha=0.3)
ax2.set_ylim(0, 1.1)

fig1.suptitle("Object Detector Evaluation: COCO YOLOv8m vs YOLO-World v2\n(no ground truth: detections and confidence, not accuracy)", fontsize=13, fontweight="bold", y=0.98)
fig1.text(0.5, 0.01, "Footnote: n = 9 style outputs; ★ highlights key architectural / furnishing targets (window, door, ladder, tv).", ha="center", fontsize=8, fontstyle="italic", color="#555555")

plt.tight_layout(rect=[0, 0.04, 1, 0.93])
fig1_path = FIGURES_DIR / "detection_metrics.png"
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)
print(f"  [1/6] Saved: {fig1_path.name}")

# ── FIGURE 2: Structure Preservation per Style ──────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 6), dpi=300)

labels = [r["display_name"] for r in results_per_style]
canny_vals = [r["canny_edge_ssim"] for r in results_per_style]
depth_vals = [r["depth_pearson_corr"] for r in results_per_style]

x = np.arange(len(labels))
width = 0.36

rects1 = ax.bar(x - width/2, canny_vals, width, label="Canny-Edge SSIM", color="#3d5a80", edgecolor="#1d2d44", linewidth=0.8)
rects2 = ax.bar(x + width/2, depth_vals, width, label="Depth Pearson Correlation (r)", color="#0081a7", edgecolor="#004e64", linewidth=0.8)

# Add horizontal mean lines
ax.axhline(mean_ssim, color="#3d5a80", linestyle=":", linewidth=1.5, alpha=0.8, label=f"Mean Edge SSIM ({mean_ssim:.3f})")
ax.axhline(mean_depth_corr, color="#0081a7", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Mean Depth Corr ({mean_depth_corr:.3f})")

# Value annotations
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9, fontweight="medium")
ax.set_ylabel("Preservation Score (SSIM / Pearson r)", fontsize=10, fontweight="bold")
ax.set_title("Structural Preservation per Style Relative to Paired Input Photo\n(Evaluated at 512 × 512 px)", fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")

fig2.text(0.5, 0.01, "Footnote: n = 9 style outputs compared against paired source photos at 512 px. Depth estimated using LiheYoung/depth-anything-small-hf.", ha="center", fontsize=8, fontstyle="italic", color="#555555")

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
fig2_path = FIGURES_DIR / "structure_preservation_per_style.png"
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)
print(f"  [2/6] Saved: {fig2_path.name}")

# ── FIGURE 3: CLIP Style Alignment per Style ────────────────────────────────
fig3, ax = plt.subplots(figsize=(12, 6), dpi=300)

clip_sims = [r["clip_target_similarity"] for r in results_per_style]
top1_flags = [r["clip_top1_match"] for r in results_per_style]

bar_colors = ["#2a9d8f" if match else "#e76f51" for match in top1_flags]
bars = ax.bar(x, clip_sims, width=0.55, color=bar_colors, edgecolor="#264653", linewidth=0.8)

ax.axhline(mean_clip_sim, color="#264653", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Mean Similarity ({mean_clip_sim:.3f})")

for bar, match, r in zip(bars, top1_flags, results_per_style):
    h = bar.get_height()
    status_str = "Top-1 ✓" if match else f"Pred: {r['clip_predicted_style'][:8]}"
    ax.annotate(f"{h:.3f}\n({status_str})", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Custom legend for Top-1 indicator
import matplotlib.patches as mpatches
top1_patch = mpatches.Patch(color="#2a9d8f", label=f"Top-1 Match (Accuracy: {top1_accuracy*100:.1f}%)")
mismatch_patch = mpatches.Patch(color="#e76f51", label="Top-1 Confused")
handles, labels_l = ax.get_legend_handles_labels()
handles.extend([top1_patch, mismatch_patch])
ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9, fontweight="medium")
ax.set_ylabel("CLIP Cosine Similarity", fontsize=10, fontweight="bold")
ax.set_title("Zero-Shot CLIP Style Alignment per Style\nTarget Prompt: 'a {style} style interior room' (CLIP ViT-B/32)", fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, max(clip_sims) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.3)

fig3.text(0.5, 0.01, f"Footnote: n = 9 style outputs; prompt template: 'a {{style}} style interior room'; Zero-Shot Top-1 Accuracy = {top1_accuracy*100:.1f}%.", ha="center", fontsize=8, fontstyle="italic", color="#555555")

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
fig3_path = FIGURES_DIR / "clip_style_alignment_per_style.png"
fig3.savefig(fig3_path, dpi=300)
plt.close(fig3)
print(f"  [3/6] Saved: {fig3_path.name}")

# ── FIGURE 4: Style Confusion Matrix ────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(10, 9), dpi=300)

im = ax.imshow(similarity_matrix_9x9, cmap="Blues", interpolation="nearest")
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("CLIP Cosine Similarity", fontsize=10, fontweight="bold")

display_names_short = [r["display_name"] for r in results_per_style]
ax.set_xticks(np.arange(9))
ax.set_yticks(np.arange(9))
ax.set_xticklabels(display_names_short, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(display_names_short, fontsize=9)

# Annotate cells with similarity value and star on top-1 prediction
thresh = (similarity_matrix_9x9.max() + similarity_matrix_9x9.min()) / 2.0
for i in range(9):
    pred = int(similarity_matrix_9x9[i].argmax())
    for j in range(9):
        val = similarity_matrix_9x9[i, j]
        is_pred = (j == pred)
        marker = " [★]" if is_pred else ""
        text_color = "white" if val > thresh else "black"
        font_weight = "bold" if is_pred else "normal"
        ax.text(j, i, f"{val:.3f}{marker}", ha="center", va="center", color=text_color,
                fontsize=8, fontweight=font_weight)

ax.set_xlabel("Candidate Style Prompt ('a {style} style interior room')", fontsize=10, fontweight="bold", labelpad=10)
ax.set_ylabel("True Generated Image Style", fontsize=10, fontweight="bold", labelpad=10)
ax.set_title(f"9 × 9 Style Similarity Matrix & Top-1 Predictions\n(★ indicates zero-shot top-1 predicted style, Accuracy = {top1_accuracy*100:.1f}%)", fontsize=12, fontweight="bold", pad=14)

fig4.text(0.5, 0.01, f"Footnote: n = 9 style outputs; rows: ground-truth style, columns: prompt candidate similarity; [★] = argmax candidate.", ha="center", fontsize=8, fontstyle="italic", color="#555555")

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
fig4_path = FIGURES_DIR / "style_confusion_matrix.png"
fig4.savefig(fig4_path, dpi=300)
plt.close(fig4)
print(f"  [4/6] Saved: {fig4_path.name}")

# ── FIGURE 5: Input vs Output Grid ──────────────────────────────────────────
fig5, axes = plt.subplots(3, 3, figsize=(15, 15), dpi=300)
axes = axes.flatten()

for idx, (ax_sub, item) in enumerate(zip(axes, STYLE_DATA)):
    im_in = Image.open(item["local_input"]).convert("RGB")
    im_out = Image.open(item["local_output"]).convert("RGB")

    # Combine side-by-side
    target_h = 400
    w_in = int(im_in.width * (target_h / im_in.height))
    w_out = int(im_out.width * (target_h / im_out.height))
    in_resized = im_in.resize((w_in, target_h), Image.Resampling.BILINEAR)
    out_resized = im_out.resize((w_out, target_h), Image.Resampling.BILINEAR)

    combined = Image.new("RGB", (w_in + w_out + 10, target_h), (255, 255, 255))
    combined.paste(in_resized, (0, 0))
    combined.paste(out_resized, (w_in + 10, 0))

    ax_sub.imshow(combined)
    ax_sub.axis("off")
    title_text = f"{item['display_name']}\nInput: {Path(item['input_source']).name}  →  Output: {Path(item['output_source']).name}"
    ax_sub.set_title(title_text, fontsize=9, fontweight="bold", pad=6)

fig5.suptitle("House Muse — Input Photo vs Generated Output Comparison (All 9 Styles)", fontsize=14, fontweight="bold", y=0.99)
fig5.text(0.5, 0.01, "Footnote: n = 9 paired images; Left: Original Uploaded Photo, Right: Generated Interior Design Output.", ha="center", fontsize=9, fontstyle="italic", color="#555555")

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
fig5_path = FIGURES_DIR / "input_vs_output_grid.png"
fig5.savefig(fig5_path, dpi=300)
plt.close(fig5)
print(f"  [5/6] Saved: {fig5_path.name}")

# ── FIGURE 6: Comprehensive Summary Table ───────────────────────────────────
fig6, ax_tbl = plt.subplots(figsize=(16, 9), dpi=300)
ax_tbl.axis("off")

table_rows = []
for r in results_per_style:
    table_rows.append([
        r["display_name"],
        r["input_source"],
        r["output_source"],
        f"{r['canny_edge_ssim']:.4f}",
        f"{r['depth_pearson_corr']:.4f}",
        f"{r['clip_target_similarity']:.4f}",
        f"{r['clip_predicted_style']} ({'✓' if r['clip_top1_match'] else '✗'})",
        f"{r['coco_detections_out_count']}",
        f"{r['world_detections_out_count']}"
    ])

# Aggregate row
table_rows.append([
    "OVERALL MEAN / ACCURACY",
    "-",
    "-",
    f"{mean_ssim:.4f}",
    f"{mean_depth_corr:.4f}",
    f"{mean_clip_sim:.4f}",
    f"Acc: {top1_accuracy*100:.1f}%",
    f"{np.mean([r['coco_detections_out_count'] for r in results_per_style]):.1f}",
    f"{np.mean([r['world_detections_out_count'] for r in results_per_style]):.1f}"
])

col_labels = [
    "Style",
    "Paired Input Photo",
    "Generated Output",
    "Canny SSIM",
    "Depth Corr (r)",
    "CLIP Sim",
    "CLIP Top-1 Pred",
    "YOLOv8m Dets",
    "YOLO-World Dets"
]

table = ax_tbl.table(
    cellText=table_rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="upper center",
    bbox=[0.0, 0.42, 1.0, 0.52]
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)

# Style header and mean row
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor("#1f4e79")
    table[(0, j)].set_text_props(color="white", fontweight="bold")
    table[(len(table_rows), j)].set_facecolor("#e2efda")
    table[(len(table_rows), j)].set_text_props(fontweight="bold")

# Add "Not Computed Metrics" section beneath the table
not_comp_text = (
    "METRICS NOT COMPUTED (AND RATIONALE):\n"
    "• Custom Detector (best.pt): Not computed — weights file 'models/checkpoints/house_muse_yolo/weights/best.pt' not found on disk.\n"
    "• Labelled Detection Dataset mAP: Not computed — no ground-truth annotations exist; evaluated via per-class detections and confidence.\n"
    "• SAM Pixel Segmentation: Not computed — segment-anything library and sam_vit_b_01ec64.pth checkpoint not on disk.\n"
    "• BLIP Image Captioning: Not computed — remote API call skipped to enforce zero GPU quota and zero external dependencies.\n"
    "• LoRA Training Loss: Not computed — training logs (trainer_state.json, TensorBoard events) not found in repo; ask teammate who trained them."
)

ax_tbl.text(0.02, 0.35, not_comp_text, fontsize=8.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.0))

ax_tbl.set_title("House Muse — Comprehensive Model Evaluation & Metric Summary", fontsize=14, fontweight="bold", y=0.98)
fig6.text(0.5, 0.02, "Footnote: n = 9 style outputs; Canny SSIM and Depth Correlation computed at 512px; CLIP alignment using ViT-B/32.", ha="center", fontsize=8.5, fontstyle="italic", color="#555555")

plt.tight_layout(rect=[0, 0.04, 1, 0.98])
fig6_path = FIGURES_DIR / "summary_table.png"
fig6.savefig(fig6_path, dpi=300)
plt.close(fig6)
print(f"  [6/6] Saved: {fig6_path.name}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. ZIP FIGURES FOLDER
# ─────────────────────────────────────────────────────────────────────────────
zip_path = EVAL_DIR / "evaluation_figures.zip"
print(f"\n[Step 5] Zipping figures to {zip_path.name}...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for fig_file in FIGURES_DIR.glob("*.png"):
        zipf.write(fig_file, arcname=fig_file.name)
print(f"Zip archive created: {zip_path} ({zip_path.stat().st_size} bytes)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. VERIFY FIGURES AND STATUS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 6] Verification:")
all_pngs = list(FIGURES_DIR.glob("*.png"))
print(f"Total figures generated: {len(all_pngs)}")
all_valid = True
for p in all_pngs:
    sz = p.stat().st_size
    valid = sz > 1000
    try:
        with Image.open(p) as img:
            img.verify()
        print(f"  ✓ {p.name:36s} | Size: {sz:8d} bytes | Verified valid PNG")
    except Exception as e:
        print(f"  ✗ {p.name:36s} | Corrupted: {e}")
        all_valid = False

if all_valid and len(all_pngs) == 6:
    print("\n✓ ALL 6 FIGURES SUCCESSFULLY GENERATED, VERIFIED AND PACKAGED!")
else:
    print("\n✗ SOME FIGURES FAILED VERIFICATION!")
