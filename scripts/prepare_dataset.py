"""
Smart Bin AI — Dataset Preparation Script
Downloads (if needed), remaps, merges, resizes, augments, and splits
three Kaggle waste-classification datasets into a unified 4-class structure.

Usage:
    python scripts/prepare_dataset.py

Expects raw datasets in:
    data/raw/trashnet/
    data/raw/garbage/
    data/raw/waste/

Produces:
    data/train/{Plastic,Paper,Metal,Glass}/
    data/val/{Plastic,Paper,Metal,Glass}/
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_ROOT = Path("data/raw")
OUTPUT_ROOT = Path("data")
TARGET_SIZE = (224, 224)
VAL_RATIO = 0.20
SEED = 42

# Mapping from raw folder names → target class
CLASS_MAP: dict[str, str | None] = {
    # TrashNet
    "cardboard": "Paper",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": None,           # skip
    # Garbage Classification (asdasdasasdas)
    "Cardboard": "Paper",
    "Glass": "Glass",
    "Metal": "Metal",
    "Paper": "Paper",
    "Plastic": "Plastic",
    "Trash": None,
    # Waste Classification (techsash)
    "ORGANIC": None,          # skip
    "O": None,
    "R": None,                # we'll handle RECYCLABLE per-image below
    "RECYCLABLE": None,       # handled separately
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create train/ and val/ directories for each target class."""
    for split in ("train", "val"):
        for cls in ("Plastic", "Paper", "Metal", "Glass"):
            (OUTPUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply random augmentations: flip, brightness jitter ±20%, rotation ±15°."""
    # Random horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Brightness jitter ±20%
    factor = random.uniform(0.80, 1.20)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)

    # Rotation ±15°
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    return img


def copy_and_process(
    src: Path,
    target_class: str,
    counter: dict[str, int],
    do_augment: bool = True,
) -> None:
    """
    Read *src* image, resize to 224×224, optionally augment, and save
    to the correct train/ or val/ split folder.
    """
    img = cv2.imread(str(src))
    if img is None:
        return
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Decide split
    split = "val" if random.random() < VAL_RATIO else "train"

    # Augment training images
    if split == "train" and do_augment:
        img = augment_image(img)

    idx = counter.get(target_class, 0)
    counter[target_class] = idx + 1

    out_name = f"{target_class.lower()}_{idx:06d}.jpg"
    out_path = OUTPUT_ROOT / split / target_class / out_name
    cv2.imwrite(str(out_path), img)


# ---------------------------------------------------------------------------
# Dataset walkers
# ---------------------------------------------------------------------------

def process_trashnet(counter: dict[str, int]) -> None:
    base = RAW_ROOT / "trashnet"
    if not base.exists():
        print(f"⚠  Skipping TrashNet — {base} not found")
        return
    # TrashNet usually extracts to dataset-resized/ or similar
    for sub in base.rglob("*"):
        if sub.is_file() and sub.suffix.lower() in (".jpg", ".jpeg", ".png"):
            folder = sub.parent.name.lower()
            cls = CLASS_MAP.get(folder) or CLASS_MAP.get(folder.capitalize())
            if cls:
                copy_and_process(sub, cls, counter)


def process_garbage(counter: dict[str, int]) -> None:
    base = RAW_ROOT / "garbage"
    if not base.exists():
        print(f"⚠  Skipping Garbage Classification — {base} not found")
        return
    for sub in base.rglob("*"):
        if sub.is_file() and sub.suffix.lower() in (".jpg", ".jpeg", ".png"):
            folder = sub.parent.name
            cls = CLASS_MAP.get(folder) or CLASS_MAP.get(folder.lower())
            if cls:
                copy_and_process(sub, cls, counter)


def process_waste(counter: dict[str, int]) -> None:
    base = RAW_ROOT / "waste"
    if not base.exists():
        print(f"⚠  Skipping Waste Classification — {base} not found")
        return
    # Only RECYCLABLE images; skip ORGANIC
    for sub in base.rglob("*"):
        if sub.is_file() and sub.suffix.lower() in (".jpg", ".jpeg", ".png"):
            # Heuristic: check parent folder names
            parts = [p.lower() for p in sub.parts]
            if "organic" in parts or "o" in parts:
                continue
            # Try to infer class from filename or parent
            name_lower = sub.stem.lower()
            cls = None
            for keyword, target in [
                ("plastic", "Plastic"),
                ("paper", "Paper"),
                ("cardboard", "Paper"),
                ("metal", "Metal"),
                ("glass", "Glass"),
            ]:
                if keyword in name_lower:
                    cls = target
                    break
            # Fallback: put recyclable unknowns into Plastic (most common)
            if cls is None:
                cls = "Plastic"
            copy_and_process(sub, cls, counter)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print("Smart Bin AI — Dataset Preparation")
    print("=" * 50)
    ensure_dirs()

    counter: dict[str, int] = {}

    print("\n[1/3] Processing TrashNet …")
    process_trashnet(counter)

    print("[2/3] Processing Garbage Classification …")
    process_garbage(counter)

    print("[3/3] Processing Waste Classification …")
    process_waste(counter)

    print("\n✅ Done!  Merged dataset stats:")
    for split in ("train", "val"):
        print(f"\n  {split}/")
        for cls in ("Plastic", "Paper", "Metal", "Glass"):
            n = len(list((OUTPUT_ROOT / split / cls).glob("*")))
            print(f"    {cls:10s}  {n:>5d} images")

    print(
        "\n➡  Upload data/train/ folders to Google Teachable Machine "
        "and export as TFLite."
    )


if __name__ == "__main__":
    main()
