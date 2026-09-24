from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from data_loader import aspect_ratio_ok, detect_blur, image_hash, preprocess

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "Medicines.v17i.coco"
SPLITS = ["train", "valid", "test"]
OUTPUT_ROOT = ROOT / "genuine_split"


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-")
    return cleaned or "unknown"


def load_coco(split: str) -> Optional[Dict]:
    annotation_path = DATASET_ROOT / split / "_annotations.coco.json"
    if not annotation_path.exists():
        print(f"[info] No annotation file found for {split}: {annotation_path}")
        return None

    try:
        with annotation_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"[warning] Unable to parse COCO annotations for {split}: {exc}")
        return None


def build_label_map(coco: Dict) -> Dict[int, str]:
    if not coco:
        return {}

    return {
        category.get("id"): sanitize_name(category.get("name", "unknown"))
        for category in coco.get("categories", [])
        if category.get("id") is not None
    }


def resolve_image_path(split_dir: Path, image_info: Dict) -> Optional[Path]:
    file_name = image_info.get("file_name", "")
    if not file_name:
        return None

    direct_path = split_dir / file_name
    if direct_path.exists():
        return direct_path

    fallback = DATASET_ROOT / split_dir.name / file_name
    if fallback.exists():
        return fallback

    return None


def clean_crop(crop: Image.Image, crop_path: Path, seen_hashes: Set[str]) -> Optional[Image.Image]:
    """Return a cleansed crop image if duplicate, blur, or bad aspect ratio filters pass."""
    try:
        processed = preprocess(crop)
        h = image_hash(processed)
        if h in seen_hashes:
            print(f"[skip] Duplicate crop hash {h} for {crop_path.name}")
            return None

        # Blur detection from variance of Laplacian.
        blur_score = detect_blur(processed)
        if blur_score < 80.0:
            print(f"[skip] Blur score too low for {crop_path.name}: {blur_score:.2f}")
            return None

        # Reject suspicious aspect ratios.
        if not aspect_ratio_ok(processed, min_ratio=0.35, max_ratio=2.5):
            print(f"[skip] Bad aspect ratio for {crop_path.name}: {processed.size}")
            return None

        seen_hashes.add(h)
        return processed
    except Exception as exc:
        print(f"[warning] Error cleaning crop {crop_path.name}: {exc}")
        return None


def crop_and_save(coco: Dict, split: str, label_map: Dict[int, str]) -> int:
    if not coco:
        return 0

    split_dir = DATASET_ROOT / split
    images = {image.get("id"): image for image in coco.get("images", []) if image.get("id") is not None}
    seen_hashes: Set[str] = set()
    saved_count = 0
    per_label_count: Dict[str, int] = {}

    for annotation in coco.get("annotations", []):
        image_info = images.get(annotation.get("image_id"))
        if image_info is None:
            continue

        image_path = resolve_image_path(split_dir, image_info)
        if image_path is None or not image_path.exists():
            print(f"[skip] Missing image for split={split}: {image_info.get('file_name')}")
            continue

        label_name = label_map.get(annotation.get("category_id"), "unknown")
        label_dir = OUTPUT_ROOT / label_name
        label_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(image_path) as image:
                image_rgb = ImageOps.exif_transpose(image).copy().convert("RGB")
                width, height = image_rgb.size
                x, y, bbox_w, bbox_h = annotation.get("bbox", [0, 0, width, height])
                x1 = max(0, int(round(x)))
                y1 = max(0, int(round(y)))
                x2 = min(width, int(round(x + bbox_w)))
                y2 = min(height, int(round(y + bbox_h)))

                if x2 <= x1 or y2 <= y1:
                    print(f"[skip] Invalid crop region for {image_path.name}")
                    continue

                cropped = image_rgb.crop((x1, y1, x2, y2))
                output_name = f"{split}_{image_info.get('id')}_{label_name}_{len(list(label_dir.glob('*.jpg')))}.jpg"
                output_path = label_dir / output_name

                cleaned = clean_crop(cropped, output_path, seen_hashes)
                if cleaned is None:
                    continue

                cleaned.save(output_path)
                saved_count += 1
                per_label_count[label_name] = per_label_count.get(label_name, 0) + 1
        except Exception as exc:
            print(f"[warning] Unable to crop {image_path.name}: {exc}")

    print(f"[{split}] saved {saved_count} crops")
    return saved_count


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Create class directories as a compatibility preservation step.
    for medicine_dir in ["Ace_XR", "Monas", "Vitabion"]:
        (OUTPUT_ROOT / medicine_dir).mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for split in SPLITS:
        coco = load_coco(split)
        if coco is None:
            continue

        label_map = build_label_map(coco)
        total_saved += crop_and_save(coco, split, label_map)

    print(f"Finished. Cropped images are stored in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
