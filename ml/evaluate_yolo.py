"""
evaluate_yolo.py
════════════════════════════════════════════════════════════════════════════════
Calculates Precision, Recall, F1, mAP@50, mAP@50-95 for your trained
YOLOv8 model on the IKEA-master validation set.

Run from inside backend/ folder:
    python evaluate_yolo.py

Requirements:
    pip install ultralytics scikit-learn matplotlib seaborn pandas
════════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed — saves plots to files
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these two paths if yours differ
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS   = "models/checkpoints/house_muse_yolo/weights/best.pt"
DATA_YAML = "data/processed/dataset.yaml"
OUT_DIR   = Path("evaluation_results")
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS NAMES  (must match training/dataset_builder.py order)
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = [
    "sofa", "chair", "bed", "dining_table", "wardrobe",
    "bookshelf", "lamp", "rug", "curtain", "plant",
    "vase", "cushion", "painting", "television",
    "coffee_table", "side_table", "mirror", "desk", "clock",
]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Run YOLOv8 built-in validation
#           This gives mAP, precision, recall per class directly
# ══════════════════════════════════════════════════════════════════════════════

def run_yolo_validation():
    print("\n" + "═"*60)
    print("  STEP 1 — YOLOv8 Validation (mAP, Precision, Recall)")
    print("═"*60)

    if not Path(WEIGHTS).exists():
        print(f"\n[ERROR] Weights not found: {WEIGHTS}")
        print("Make sure training completed and best.pt exists.")
        return None

    model   = YOLO(WEIGHTS)
    metrics = model.val(
        data      = DATA_YAML,
        imgsz     = 640,
        conf      = 0.25,          # confidence threshold
        iou       = 0.50,          # IoU threshold for match
        device    = "cpu",
        verbose   = True,
        save_json = True,
        project   = str(OUT_DIR),
        name      = "val_run",
    )

    # ── Extract core metrics ──────────────────────────────────────────────────
    map50    = round(float(metrics.box.map50),   4)
    map5095  = round(float(metrics.box.map),     4)
    prec     = round(float(metrics.box.mp),      4)
    recall   = round(float(metrics.box.mr),      4)

    # F1 score = harmonic mean of precision and recall
    if prec + recall > 0:
        f1 = round(2 * prec * recall / (prec + recall), 4)
    else:
        f1 = 0.0

    print("\n" + "─"*60)
    print(f"  mAP@50        : {map50}   (target > 0.70)")
    print(f"  mAP@50-95     : {map5095} (target > 0.50)")
    print(f"  Precision     : {prec}    (TP / (TP+FP))")
    print(f"  Recall        : {recall}  (TP / (TP+FN))")
    print(f"  F1 Score      : {f1}      (harmonic mean)")
    print("─"*60)

    return metrics, {"map50":map50, "map5095":map5095,
                     "precision":prec, "recall":recall, "f1":f1}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Per-class precision, recall, F1
# ══════════════════════════════════════════════════════════════════════════════

def per_class_metrics(metrics):
    print("\n" + "═"*60)
    print("  STEP 2 — Per-class Precision / Recall / F1")
    print("═"*60)

    # metrics.box.ap_class_index → class indices that were evaluated
    # metrics.box.p  → precision per class
    # metrics.box.r  → recall per class
    # metrics.box.ap50 → AP@50 per class

    try:
        class_indices = metrics.box.ap_class_index.tolist()
        p_vals   = metrics.box.p.tolist()
        r_vals   = metrics.box.r.tolist()
        ap50_vals= metrics.box.ap50.tolist()
    except Exception as e:
        print(f"[WARN] Could not extract per-class metrics: {e}")
        return pd.DataFrame()

    rows = []
    for i, cls_idx in enumerate(class_indices):
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
        p  = round(p_vals[i], 4)   if i < len(p_vals)    else 0.0
        r  = round(r_vals[i], 4)   if i < len(r_vals)    else 0.0
        ap = round(ap50_vals[i], 4) if i < len(ap50_vals) else 0.0
        f1 = round(2*p*r/(p+r), 4) if (p+r) > 0 else 0.0
        rows.append({"Class": cls_name, "Precision": p,
                     "Recall": r, "F1": f1, "AP@50": ap})

    df = pd.DataFrame(rows).sort_values("AP@50", ascending=False)

    print("\n" + df.to_string(index=False))

    # Save to CSV
    csv_path = OUT_DIR / "per_class_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Confusion matrix (what gets confused with what)
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(metrics, df_classes):
    print("\n" + "═"*60)
    print("  STEP 3 — Confusion Matrix")
    print("═"*60)

    try:
        cm     = metrics.confusion_matrix.matrix    # numpy array (nc+1, nc+1)
        n      = cm.shape[0]
        # Last row/col is background — trim if all-zero
        labels = CLASS_NAMES[:n] if n <= len(CLASS_NAMES) else CLASS_NAMES

        # Normalise by row (ground-truth class totals)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm, row_sums, where=row_sums!=0) * 100

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm_norm, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.3, linecolor="white",
            annot_kws={"size": 7}, ax=ax
        )
        ax.set_title("Confusion Matrix — normalised (%) \nYOLOv8 on IKEA-master val set",
                     fontsize=13, pad=14)
        ax.set_xlabel("Predicted class", fontsize=11)
        ax.set_ylabel("True class",      fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0,  fontsize=8)
        plt.tight_layout()

        out = OUT_DIR / "confusion_matrix.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved → {out}")

    except Exception as e:
        print(f"  [WARN] Could not plot confusion matrix: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Precision-Recall curve (visual)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pr_curve(df_classes):
    print("\n" + "═"*60)
    print("  STEP 4 — Precision vs Recall bar chart (per class)")
    print("═"*60)

    if df_classes.empty:
        print("  [SKIP] No per-class data available.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    x      = np.arange(len(df_classes))
    width  = 0.28

    ax.bar(x - width, df_classes["Precision"], width, label="Precision", color="#3B6D11", alpha=0.85)
    ax.bar(x,          df_classes["Recall"],   width, label="Recall",    color="#185FA5", alpha=0.85)
    ax.bar(x + width,  df_classes["F1"],       width, label="F1 Score",  color="#854F0B", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df_classes["Class"], rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 per class — YOLOv8 IKEA-master", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = OUT_DIR / "precision_recall_f1_per_class.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — mAP@50 bar chart (sorted best to worst class)
# ══════════════════════════════════════════════════════════════════════════════

def plot_map_per_class(df_classes):
    print("\n" + "═"*60)
    print("  STEP 5 — AP@50 per class")
    print("═"*60)

    if df_classes.empty:
        return

    df_sorted = df_classes.sort_values("AP@50", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(5, len(df_sorted)*0.45)))
    bars    = ax.barh(df_sorted["Class"], df_sorted["AP@50"],
                      color="#3B8BD4", alpha=0.85, edgecolor="white", linewidth=0.4)

    # Annotate values
    for bar, val in zip(bars, df_sorted["AP@50"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("AP@50")
    ax.set_title("Average Precision @IoU=0.50 per class", fontsize=12)
    ax.axvline(x=0.70, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Target (0.70)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = OUT_DIR / "map50_per_class.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Overall summary printout + JSON
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(overall, df_classes):
    print("\n" + "═"*60)
    print("  STEP 6 — Final Summary")
    print("═"*60)

    # Interpretation
    map50 = overall["map50"]
    if map50 >= 0.80:
        grade = "Excellent — ready for production"
    elif map50 >= 0.70:
        grade = "Good — suitable for demo/project"
    elif map50 >= 0.55:
        grade = "Fair — needs more training epochs or data"
    else:
        grade = "Needs improvement — check dataset labels"

    summary = {
        "model_weights":  WEIGHTS,
        "dataset":        DATA_YAML,
        "overall_metrics": overall,
        "grade":          grade,
        "per_class":      df_classes.to_dict("records") if not df_classes.empty else [],
    }

    json_path = OUT_DIR / "evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"""
  ┌─────────────────────────────────────────┐
  │  Model       : {Path(WEIGHTS).name:<25} │
  │  mAP@50      : {overall['map50']:<25} │
  │  mAP@50-95   : {overall['map5095']:<25} │
  │  Precision   : {overall['precision']:<25} │
  │  Recall      : {overall['recall']:<25} │
  │  F1 Score    : {overall['f1']:<25} │
  │  Grade       : {grade:<25} │
  └─────────────────────────────────────────┘
    """)
    print(f"  Full summary → {json_path}")
    print(f"  Outputs in   → {OUT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Quick snippet: manual precision/recall/F1 using sklearn
#           (alternative calculation to double-check)
# ══════════════════════════════════════════════════════════════════════════════

def manual_sklearn_check():
    """
    If you have prediction .txt files and ground truth .txt files (YOLO format),
    this snippet shows how to compute metrics using sklearn.
    This is a demonstration — it requires you to collect predictions first.
    """
    print("\n" + "═"*60)
    print("  STEP 7 — Manual sklearn snippet (demonstration)")
    print("═"*60)

    # Example: simulated ground truth vs predictions for 3 classes
    # Replace with real data from your val set predictions

    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )

    # Simulate: 50 ground truth labels, predicted labels
    # 0=sofa, 1=chair, 2=bed (simplified 3-class example)
    np.random.seed(42)
    y_true = np.random.choice([0, 1, 2], size=100,
                               p=[0.4, 0.35, 0.25])   # ground truth class labels
    # Simulate ~80% accurate predictions
    y_pred = y_true.copy()
    noise_idx = np.random.choice(len(y_true), size=18, replace=False)
    y_pred[noise_idx] = np.random.choice([0, 1, 2], size=18)

    p  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r  = recall_score(   y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(       y_true, y_pred, average="macro", zero_division=0)

    print(f"""
  Using sklearn — macro-averaged over 3 classes (demo only):
  ┌───────────────────────────────┐
  │  Precision  : {p:.4f}          │
  │  Recall     : {r:.4f}          │
  │  F1 Score   : {f1:.4f}          │
  └───────────────────────────────┘

  Full classification report:
  """)
    print(classification_report(y_true, y_pred,
                                 target_names=["sofa","chair","bed"],
                                 zero_division=0))

    print("""
  ── What each metric means for your project ──────────────────

  Precision = Of all the times the model said "this is a sofa",
              how often was it actually a sofa?
              → High precision = few false positives

  Recall    = Of all the real sofas in the images,
              how many did the model find?
              → High recall = few missed objects

  F1 Score  = Harmonic mean of precision and recall.
              Useful when dataset is imbalanced (more sofas than clocks).
              → Best single number to quote in your project report

  mAP@50    = Mean Average Precision at IoU threshold 0.5
              = average of AP scores across all 19 classes
              → The industry-standard metric for object detectors
              → Quote this as your primary accuracy metric
  """)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║   House Muse — YOLOv8 Evaluation Script                 ║
║   Model trained on IKEA-master + ADE20K dataset         ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Check weights exist
    if not Path(WEIGHTS).exists():
        print(f"[ERROR] Weights file not found: {WEIGHTS}")
        print("Possible paths to check:")
        for p in Path("models").rglob("best.pt"):
            print(f"  → {p}")
        exit(1)

    # Run all steps
    result      = run_yolo_validation()
    if result is None:
        exit(1)

    metrics, overall = result
    df_classes       = per_class_metrics(metrics)

    plot_confusion_matrix(metrics, df_classes)
    plot_pr_curve(df_classes)
    plot_map_per_class(df_classes)
    save_summary(overall, df_classes)
    manual_sklearn_check()

    print("\n  Done. Open evaluation_results/ to see all plots and CSV.\n")
