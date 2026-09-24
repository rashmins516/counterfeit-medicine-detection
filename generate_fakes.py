from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent
GENUINE_ROOT = ROOT / "genuine_split"
FAKES_ROOT = ROOT / "fakes"


def apply_counterfeit_distortions(img_np: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply a family of reproducible counterfeit-style distortions.

    Distortion families are kept explicit and deterministic in seeding when needed,
    making fake generation reproducible and inspectable by a human reviewer.
    """
    rng = np.random.default_rng()
    distorted = img_np.copy().astype(np.uint8)
    h, w, _ = distorted.shape

    # 1. Color shift / ink mismatch.
    if random.random() < 0.75 * strength:
        hsv = cv2.cvtColor(distorted, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-20, 20)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.4), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
        distorted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 2. Print blur and sharpness variations.
    if random.random() < 0.65 * strength:
        ksize = random.choice([3, 5, 7])
        distorted = cv2.GaussianBlur(distorted, (ksize, ksize), 0)

    # 3. Speckle noise and ink splatter.
    if random.random() < 0.55 * strength:
        noise = np.random.normal(0, random.uniform(8, 22), distorted.shape)
        distorted = np.clip(distorted.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Low contrast / faded print.
    if random.random() < 0.5 * strength:
        alpha = random.uniform(0.7, 0.95)
        beta = random.randint(-22, 18)
        distorted = cv2.convertScaleAbs(distorted, alpha=alpha, beta=beta)

    # 5. Localization distortion (logo, hologram, stamp).
    if random.random() < 0.33 * strength:
        rw = max(8, int(random.uniform(0.2, 0.45) * w))
        rh = max(8, int(random.uniform(0.2, 0.45) * h))
        rx = random.randint(0, max(0, w - rw))
        ry = random.randint(0, max(0, h - rh))
        roi = distorted[ry : ry + rh, rx : rx + rw]
        roi = cv2.resize(roi, (max(1, rw // 4), max(1, rh // 4)), interpolation=cv2.INTER_NEAREST)
        roi = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_NEAREST)
        distorted[ry : ry + rh, rx : rx + rw] = roi

    return distorted


def balanced_fake_set_sizes(sku_dirs: Sequence[Path]) -> dict[str, int]:
    """Create a deterministic unique set of fake counts per SKU to keep class balance.

    Returns a per-folder fake count map.
    """
    counts: dict[str, int] = {}
    for sku_dir in sku_dirs:
        sample_count = len(list(sku_dir.glob("*.jpg"))) + len(list(sku_dir.glob("*.png")))
        counts[sku_dir.name] = max(1, sample_count)
    return counts


def generate_fake_dataset(num_fakes_per_genuine: int = 2, strength: float = 1.0) -> int:
    """Generate balanced fake crops from the cleaned genuine_split directory."""
    if num_fakes_per_genuine <= 0:
        raise ValueError("num_fakes_per_genuine must be >= 1")

    FAKES_ROOT.mkdir(parents=True, exist_ok=True)

    sku_dirs = [d for d in GENUINE_ROOT.iterdir() if d.is_dir()]
    if not sku_dirs:
        raise FileNotFoundError(f"No genuine SKU directories found under {GENUINE_ROOT}")

    counts = balanced_fake_set_sizes(sku_dirs)
    total_generated = 0

    for sku_dir in sku_dirs:
        fake_sku_dir = FAKES_ROOT / sku_dir.name
        fake_sku_dir.mkdir(parents=True, exist_ok=True)

        genuine_images = sorted(sku_dir.glob("*.jpg")) + sorted(sku_dir.glob("*.png"))
        if not genuine_images:
            continue

        for image_file in genuine_images:
            try:
                with Image.open(image_file) as img:
                    img_rgb = np.asarray(img.convert("RGB"))

                for idx in range(num_fakes_per_genuine):
                    fake_np = apply_counterfeit_distortions(img_rgb.copy(), strength=strength)
                    fake_img = Image.fromarray(fake_np.astype("uint8"))
                    output_name = f"fake_{idx + 1}_{image_file.name}"
                    fake_img.save(fake_sku_dir / output_name)
                    total_generated += 1
            except Exception as exc:
                print(f"[warning] Error processing {image_file}: {exc}")

    print(f"Generated {total_generated} fake crops across {len(sku_dirs)} SKU directories in {FAKES_ROOT}")
    return total_generated


if __name__ == "__main__":
    generate_fake_dataset(num_fakes_per_genuine=2, strength=1.0)
