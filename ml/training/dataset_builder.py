"""
training/dataset_builder.py
Converts your IKEA-master dataset into YOLO training format.
Your IKEA folder structure:
    data/raw/ikea/images/
        bed/
        chair/
        clock/
        couch/
        dining table/
        objects/
        plant_pot/
        room_scenes/
"""

import os
import json
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# ── Class list — matches image_processor.py exactly ──────────────────────────
HOUSE_MUSE_CLASSES = [
    "sofa",        # 0  ← couch maps here
    "chair",       # 1
    "bed",         # 2
    "dining_table",# 3
    "wardrobe",    # 4
    "bookshelf",   # 5
    "lamp",        # 6
    "rug",         # 7
    "curtain",     # 8
    "plant",       # 9  ← plant_pot maps here
    "vase",        # 10
    "cushion",     # 11
    "painting",    # 12
    "television",  # 13
    "coffee_table",# 14
    "side_table",  # 15
    "mirror",      # 16
    "desk",        # 17
    "clock",       # 18 ← your clock folder
]

CLASS_TO_IDX = {c: i for i, c in enumerate(HOUSE_MUSE_CLASSES)}

# ── YOUR exact IKEA folder names → House Muse class ──────────────────────────
IKEA_FOLDER_MAP = {
    "bed":          "bed",
    "chair":        "chair",
    "clock":        "clock",
    "couch":        "sofa",          # couch = sofa
    "dining table": "dining_table",
    "plant_pot":    "plant",         # plant_pot = plant
    # room_scenes and objects are skipped — no single object label
}


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    # Clamp to [0,1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.01, min(1.0, w))
    h  = max(0.01, min(1.0, h))
    return cx, cy, w, h


def auto_bbox(img: np.ndarray):
    """
    For white/light-background IKEA product photos,
    auto-detect the object bounding box by removing the background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Use largest contour
    largest = max(cnts, key=cv2.contourArea)

    # Skip if too small (< 5% of image)
    ih, iw = img.shape[:2]
    if cv2.contourArea(largest) < 0.05 * iw * ih:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    pad = int(0.04 * max(w, h))
    return (
        max(0, x - pad),
        max(0, y - pad),
        min(iw, x + w + pad),
        min(ih, y + h + pad),
    )


def write_yolo_label(path: Path, annotations: list):
    with open(path, "w") as f:
        for cls, cx, cy, w, h in annotations:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def process_ikea(raw_dir: Path, out_dir: Path, split_ratio=0.85):
    ikea_img_dir = raw_dir / "ikea" / "images"

    if not ikea_img_dir.exists():
        print(f"[ERROR] IKEA images not found at: {ikea_img_dir}")
        print(f"        Make sure you copied the dataset to: {ikea_img_dir}")
        return 0

    total = 0
    skipped = 0

    for folder_name, hm_label in IKEA_FOLDER_MAP.items():
        folder = ikea_img_dir / folder_name
        if not folder.exists():
            print(f"[WARN] Folder not found, skipping: {folder}")
            continue

        class_idx = CLASS_TO_IDX.get(hm_label)
        if class_idx is None:
            print(f"[WARN] No class index for label: {hm_label}")
            continue

        images = list(folder.glob("*.jpg")) + \
                 list(folder.glob("*.jpeg")) + \
                 list(folder.glob("*.png")) + \
                 list(folder.glob("*.webp"))

        if not images:
            print(f"[WARN] No images in: {folder}")
            continue

        print(f"[IKEA] Processing '{folder_name}' → '{hm_label}' ({len(images)} images)")

        for img_path in tqdm(images, desc=f"  {folder_name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            ih, iw = img.shape[:2]

            # Try auto bbox, fall back to full image
            bbox = auto_bbox(img)
            if bbox is None:
                # Use full image with small margin
                margin = int(0.05 * min(iw, ih))
                bbox = (margin, margin, iw - margin, ih - margin)

            x1, y1, x2, y2 = bbox
            cx, cy, bw, bh = bbox_to_yolo(x1, y1, x2, y2, iw, ih)

            split = "train" if random.random() < split_ratio else "val"

            # Save image
            out_img = out_dir / split / "images" / f"ikea_{folder_name}_{total:06d}.jpg"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img), img)

            # Save label
            out_lbl = out_dir / split / "labels" / f"ikea_{folder_name}_{total:06d}.txt"
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            write_yolo_label(out_lbl, [(class_idx, cx, cy, bw, bh)])

            total += 1

    print(f"[IKEA] Done — {total} images processed, {skipped} skipped")
    return total


def write_yaml(out_dir: Path):
    yaml_path = out_dir / "dataset.yaml"
    content = f"""# House Muse Custom Dataset
path: {out_dir.resolve().as_posix()}
train: train/images
val: val/images

nc: {len(HOUSE_MUSE_CLASSES)}
names: {HOUSE_MUSE_CLASSES}
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"[Config] dataset.yaml → {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",    default="data/raw",       help="Path to raw data folder")
    parser.add_argument("--output", default="data/processed", help="Output YOLO dataset folder")
    parser.add_argument("--seed",   default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    raw_dir = Path(args.raw)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n══ House Muse — Dataset Builder ══\n")
    total = process_ikea(raw_dir, out_dir)

    if total == 0:
        print("\n[ERROR] No images processed. Check your folder paths above.")
        return

    yaml_path = write_yaml(out_dir)

    # Print summary
    train_imgs = list((out_dir / "train" / "images").glob("*.jpg"))
    val_imgs   = list((out_dir / "val"   / "images").glob("*.jpg"))

    print(f"""
══ Dataset Ready ══
  Total images : {total}
  Train        : {len(train_imgs)}
  Val          : {len(val_imgs)}
  Config       : {yaml_path}

Next step:
  python training/train_yolo.py --data {yaml_path}
""")


if __name__ == "__main__":
    main()
