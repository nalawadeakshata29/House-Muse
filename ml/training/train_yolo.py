"""
training/train_yolo.py
════════════════════════════════════════════════════════════════════════════════
Fine-tunes YOLOv8m on the House Muse interior dataset
(IKEA furniture + ADE20K indoor scenes).

Starting from COCO pretrained YOLOv8m weights gives us:
  - All general object detection capability already learned
  - Fine-tuning adds furniture-specific accuracy + Indian/IKEA product shapes
  - Training time: ~4–6 hours on a T4 GPU (Colab/Kaggle free tier)

Run:
    python training/train_yolo.py \
        --data data/processed/dataset.yaml \
        --epochs 50 \
        --batch 16 \
        --device 0          # GPU; use "cpu" if no GPU

Output:
    models/checkpoints/house_muse_yolo/
        weights/best.pt     ← use this in image_processor.py
        weights/last.pt
        results.csv
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch


# ── Training hyperparameters ──────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Model
    "model":        "yolov8m.pt",       # medium — best speed/accuracy tradeoff
                                         # alternatives: yolov8s.pt (faster), yolov8l.pt (more accurate)
    # Training
    "epochs":       50,
    "batch":        16,
    "imgsz":        640,
    "patience":     10,                  # early stopping if no improvement
    "workers":      4,

    # Optimiser
    "optimizer":    "AdamW",
    "lr0":          0.001,
    "lrf":          0.01,                # final LR = lr0 * lrf
    "momentum":     0.937,
    "weight_decay": 0.0005,
    "warmup_epochs":3,

    # Augmentation (YOLOv8 built-in)
    "hsv_h":        0.015,               # hue augmentation (important for colour accuracy)
    "hsv_s":        0.7,
    "hsv_v":        0.4,
    "flipud":       0.0,
    "fliplr":       0.5,
    "mosaic":       1.0,                 # mosaic augmentation (mixes 4 images)
    "mixup":        0.1,
    "copy_paste":   0.1,

    # Output
    "project":      "models/checkpoints",
    "name":         "house_muse_yolo",
    "exist_ok":     True,
    "save_period":  10,                  # save checkpoint every N epochs
    "plots":        True,
    "val":          True,
}


def train(data_yaml: str, config: dict, device: str):
    """
    Fine-tune YOLOv8 on the House Muse dataset.

    Strategy:
      Phase 1 (epochs 1-10):  Freeze backbone, train head only
                               → adapt detection heads to our 18 classes
      Phase 2 (epochs 11-50): Unfreeze all layers, full fine-tuning
                               → adapt features to indoor/furniture domain
    """
    print("\n══ House Muse — YOLOv8 Training ══")
    print(f"  Data:   {data_yaml}")
    print(f"  Device: {device}")
    print(f"  Model:  {config['model']}")

    # ── Phase 1: Head-only training (frozen backbone) ──────────────────────
    print("\n[Phase 1] Training detection head only (backbone frozen) …")
    model = YOLO(config["model"])

    model.train(
        data        = data_yaml,
        epochs      = 10,
        batch       = config["batch"],
        imgsz       = config["imgsz"],
        device      = device,
        freeze      = 10,               # freeze first 10 backbone layers
        lr0         = config["lr0"] * 10,  # higher LR for head-only phase
        project     = config["project"],
        name        = config["name"] + "_phase1",
        exist_ok    = True,
        workers     = config["workers"],
        hsv_h       = config["hsv_h"],
        hsv_s       = config["hsv_s"],
        hsv_v       = config["hsv_v"],
        fliplr      = config["fliplr"],
        mosaic      = 0.5,
        verbose     = True,
    )

    # YOLO saves outputs under runs/detect/<project>/<name>/ when run from
    # the backend folder.  Build the path by checking both possibilities so
    # the script works whether you run it from backend/ or training/.
    _backend_dir = Path(__file__).parent.parent   # …/house-muse/backend/
    _phase1_name = config["name"] + "_phase1"

    phase1_weights_candidates = [
        # YOLO default: runs/detect/<project>/<name>/weights/best.pt
        _backend_dir / "runs" / "detect" / config["project"] / _phase1_name / "weights" / "best.pt",
        # Flat project path (no runs/detect prefix) – some YOLO versions / cwd combos
        _backend_dir / config["project"] / _phase1_name / "weights" / "best.pt",
        # Relative fallback (original behaviour)
        Path(config["project"]) / _phase1_name / "weights" / "best.pt",
    ]

    phase1_weights = None
    for candidate in phase1_weights_candidates:
        if candidate.exists():
            phase1_weights = candidate
            break

    if phase1_weights is None:
        searched = "\n    ".join(str(c) for c in phase1_weights_candidates)
        raise FileNotFoundError(
            f"Phase 1 weights not found. Searched:\n    {searched}\n"
            "Make sure Phase 1 training completed successfully."
        )

    print(f"[Phase 2] Loading Phase 1 weights from: {phase1_weights}")

    # ── Phase 2: Full fine-tuning ──────────────────────────────────────────
    print("\n[Phase 2] Full fine-tuning (all layers) …")
    model2 = YOLO(str(phase1_weights))

    results = model2.train(
        data            = data_yaml,
        epochs          = config["epochs"],
        batch           = config["batch"],
        imgsz           = config["imgsz"],
        device          = device,
        optimizer       = config["optimizer"],
        lr0             = config["lr0"],
        lrf             = config["lrf"],
        momentum        = config["momentum"],
        weight_decay    = config["weight_decay"],
        warmup_epochs   = config["warmup_epochs"],
        patience        = config["patience"],
        hsv_h           = config["hsv_h"],
        hsv_s           = config["hsv_s"],
        hsv_v           = config["hsv_v"],
        flipud          = config["flipud"],
        fliplr          = config["fliplr"],
        mosaic          = config["mosaic"],
        mixup           = config["mixup"],
        copy_paste      = config["copy_paste"],
        project         = config["project"],
        name            = config["name"],
        exist_ok        = config["exist_ok"],
        save_period     = config["save_period"],
        plots           = config["plots"],
        val             = config["val"],
        workers         = config["workers"],
        verbose         = True,
    )

    best_weights = Path(config["project"]) / config["name"] / "weights" / "best.pt"
    print(f"\n✓ Training complete!")
    print(f"  Best weights → {best_weights}")
    print(f"\nNext step — update image_processor.py:")
    print(f'  YOLO_WEIGHTS = "{best_weights}"')

    return results


def evaluate(weights_path: str, data_yaml: str, device: str):
    """Run validation and print mAP metrics."""
    model   = YOLO(weights_path)
    metrics = model.val(data=data_yaml, device=device)

    print(f"\n── Evaluation Results ──")
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")

    return metrics


def export_model(weights_path: str):
    """Export to ONNX for faster CPU inference (optional)."""
    model = YOLO(weights_path)
    model.export(format="onnx", dynamic=True, simplify=True)
    print(f"✓ Exported to ONNX: {weights_path.replace('.pt', '.onnx')}")


def main():
    parser = argparse.ArgumentParser(description="Train House Muse YOLOv8")
    parser.add_argument("--data",    required=True,        help="Path to dataset.yaml")
    parser.add_argument("--epochs",  default=50, type=int, help="Training epochs (Phase 2)")
    parser.add_argument("--batch",   default=16, type=int)
    parser.add_argument("--device",  default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model",   default="yolov8m.pt", help="YOLOv8 variant")
    parser.add_argument("--eval",    action="store_true",  help="Run evaluation only")
    parser.add_argument("--weights", default=None,         help="Weights for eval mode")
    parser.add_argument("--export",  action="store_true",  help="Export to ONNX after training")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, "epochs": args.epochs, "batch": args.batch, "model": args.model}

    if args.eval:
        if not args.weights:
            raise ValueError("--weights required for eval mode")
        evaluate(args.weights, args.data, args.device)
    else:
        results = train(args.data, config, args.device)
        best = f"{config['project']}/{config['name']}/weights/best.pt"
        if args.export:
            export_model(best)


if __name__ == "__main__":
    main()