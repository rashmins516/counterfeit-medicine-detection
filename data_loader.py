from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

TensorOrImage = Union[Image.Image, np.ndarray, Path, str]


def coerce_image(image: TensorOrImage) -> Image.Image:
    """Normalize a path, PIL image, or ndarray to a safe RGB PIL image."""
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            img = Image.fromarray(image.astype(np.uint8), mode="L")
        elif image.ndim == 3:
            img = Image.fromarray(image.astype(np.uint8))
        else:
            raise ValueError("Unsupported ndarray shape; expected HxW or HxWxC")
    elif isinstance(image, (str, Path)):
        path = Path(image).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as reader:
            img = ImageOps.exif_transpose(reader).copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def preprocess(image: TensorOrImage, target_size: Tuple[int, int] = (256, 256)) -> Image.Image:
    """Apply a deterministic, reusable preprocessing pipeline for crops and inference.

    Steps:
    1. Coerce inputs to RGB PIL Image.
    2. Remove EXIF orientation.
    3. Resize to a fixed target size using LANCZOS.
    4. Improve contrast and brightness slightly.
    5. Return a stable RGB image object.
    """
    img = coerce_image(image)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    # Preserve geometric consistency: fixed pixel target for training/inference.
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Slight contrast/brightness normalization keeps lighting variation controlled.
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Sharpness(img).enhance(1.05)

    return img


def detect_blur(image: Image.Image) -> float:
    """Return a blur score using the variance of Laplacian. Lower = blurrier."""
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    score = cv2.Laplacian(arr, cv2.CV_32F).var()
    return float(score)


def aspect_ratio_ok(image: Image.Image, min_ratio: float = 0.5, max_ratio: float = 2.0) -> bool:
    """Filter out highly distorted crops with suspicious aspect ratios."""
    width, height = image.size
    if height == 0 or width == 0:
        return False
    ratio = width / height
    return min_ratio <= ratio <= max_ratio


def image_hash(image: Image.Image) -> str:
    """Return a deterministic SHA-256 digest for a crop image, used to remove duplicates."""
    arr = np.asarray(image.convert("RGB"))
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return digest


def load_image_dataset(
    dataset_root: Path | str,
    max_samples_per_class: int = 200,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[List[Image.Image], List[str], List[str]]:
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
        image_files: List[Path] = []
        image_files.extend(sorted(class_dir.glob("*.jpg")))
        image_files.extend(sorted(class_dir.glob("*.jpeg")))
        image_files.extend(sorted(class_dir.glob("*.png")))
        image_files = image_files[:max_samples_per_class]

        for image_file in image_files:
            try:
                with Image.open(image_file) as raw:
                    rgb_image = ImageOps.exif_transpose(raw).copy()
                    rgb_image = rgb_image.convert("RGB")
                    samples.append(preprocess(rgb_image, target_size=target_size))
                    labels.append(class_dir.name)
            except Exception as exc:
                print(f"[warning] Skipping unreadable image {image_file}: {exc}")

    if not samples:
        raise ValueError(f"No readable images found under {dataset_path}")

    return samples, labels, class_names


def load_counterfeit_dataset(
    genuine_root: Path | str,
    fake_root: Path | str,
    max_samples_per_class: int = 200,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[List[Image.Image], List[str], List[str]]:
    """Return a combined genuine-vs-fake dataset for training or evaluation.

    The labels are class names, e.g. SKU names for the genuine source and fake labels.
    """
    genuine_path = Path(genuine_root).resolve()
    fake_path = Path(fake_root).resolve()

    if not genuine_path.exists():
        raise FileNotFoundError(f"Genuine dataset root not found: {genuine_path}")
    if not fake_path.exists():
        raise FileNotFoundError(f"Fake dataset root not found: {fake_path}")

    samples: List[Image.Image] = []
    labels: List[str] = []
    class_names: List[str] = []

    seen: set[str] = set()
    for root in (genuine_path, fake_path):
        for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            if class_dir.name not in class_names:
                class_names.append(class_dir.name)

            files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.jpeg")) + sorted(class_dir.glob("*.png"))
            cut = files[:max_samples_per_class]

            for image_file in cut:
                key = str(image_file)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    with Image.open(image_file) as raw:
                        img = ImageOps.exif_transpose(raw).copy().convert("RGB")
                        samples.append(preprocess(img, target_size=target_size))
                        labels.append(class_dir.name)
                except Exception as exc:
                    print(f"[warning] Skipping bad counterfeit sample {image_file}: {exc}")

    if not samples:
        raise ValueError("No images could be loaded from genuine/fake roots")

    return samples, labels, class_names

