from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image


def load_image_dataset(
    dataset_root: Path | str, max_samples_per_class: int = 200
) -> Tuple[List[Image.Image], List[str], List[str]]:
    """Loads images from subdirectories named after classes."""
    dataset_path = Path(dataset_root).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_path}")

    class_dirs = sorted([p for p in dataset_path.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class directories found under {dataset_path}")

    samples: List[Image.Image] = []
    labels: List[str] = []
    class_names: List[str] = [p.name for p in class_dirs]

    for class_dir in class_dirs:
        image_files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.jpeg")) + sorted(class_dir.glob("*.png"))
        image_files = image_files[:max_samples_per_class]

        for image_file in image_files:
            try:
                with Image.open(image_file) as image:
                    rgb_image = image.convert("RGB")
                    samples.append(rgb_image.copy())
                    labels.append(class_dir.name)
            except Exception:
                continue

    return samples, labels, class_names


def load_counterfeit_dataset(
    genuine_root: Path | str, fake_root: Path | str, max_samples_per_class: int = 100
) -> Tuple[List[Image.Image], List[str], List[int]]:
    """
    Loads samples from genuine and fake roots.
    Returns: (images, sku_labels, is_genuine_flags) where 1=Genuine, 0=Counterfeit.
    """
    images: List[Image.Image] = []
    sku_labels: List[str] = []
    is_genuine_flags: List[int] = []

    # Load Genuine
    gen_samples, gen_labels, _ = load_image_dataset(genuine_root, max_samples_per_class=max_samples_per_class)
    for img, lbl in zip(gen_samples, gen_labels):
        images.append(img)
        sku_labels.append(lbl)
        is_genuine_flags.append(1)

    # Load Fakes if available
    fake_path = Path(fake_root).resolve()
    if fake_path.exists() and any(fake_path.iterdir()):
        try:
            fake_samples, fake_labels, _ = load_image_dataset(fake_path, max_samples_per_class=max_samples_per_class)
            for img, lbl in zip(fake_samples, fake_labels):
                images.append(img)
                sku_labels.append(lbl)
                is_genuine_flags.append(0)
        except ValueError:
            pass

    return images, sku_labels, is_genuine_flags

