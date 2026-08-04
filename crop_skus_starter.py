from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

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

    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def crop_and_save(coco: Dict, split: str, label_map: Dict[int, str]) -> int:
    if not coco:
        return 0

    split_dir = DATASET_ROOT / split
    images = {image.get("id"): image for image in coco.get("images", []) if image.get("id") is not None}

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

        per_label_count[label_name] = per_label_count.get(label_name, 0) + 1

        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
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
            output_name = f"{split}_{image_info.get('id')}_{per_label_count[label_name]}.jpg"
            output_path = label_dir / output_name
            cropped.save(output_path)
            saved_count += 1

    print(f"[{split}] saved {saved_count} crops")
    return saved_count


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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
