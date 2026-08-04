from __future__ import annotations

import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent
GENUINE_ROOT = ROOT / "genuine_split"
FAKES_ROOT = ROOT / "fakes"


def apply_counterfeit_distortions(img_np: np.ndarray) -> np.ndarray:
    """Applies realistic physical/print packaging defects to simulate counterfeit items."""
    distorted = img_np.copy()
    h, w, _ = distorted.shape

    # 1. Color shift / Ink mismatch (change hue/saturation)
    if random.random() < 0.7:
        hsv = cv2.cvtColor(distorted, cv2.COLOR_RGB2HSV).astype(np.float32)
        h_shift = random.uniform(-15, 15)
        s_scale = random.uniform(0.7, 1.3)
        v_scale = random.uniform(0.8, 1.2)
        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
        distorted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 2. Print blur / Low resolution print
    if random.random() < 0.6:
        ksize = random.choice([3, 5, 7])
        distorted = cv2.GaussianBlur(distorted, (ksize, ksize), 0)

    # 3. Print noise / Ink splatter / Speckles
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(10, 25), distorted.shape).astype(np.float32)
        distorted = np.clip(distorted.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Poor contrast / Faded print
    if random.random() < 0.5:
        alpha = random.uniform(0.6, 0.9)  # Contrast control
        beta = random.randint(-20, 20)      # Brightness control
        distorted = cv2.convertScaleAbs(distorted, alpha=alpha, beta=beta)

    # 5. Localized distortion (simulating bad hologram or logo print defect)
    if random.random() < 0.4:
        rw = random.randint(int(w * 0.2), int(w * 0.5))
        rh = random.randint(int(h * 0.2), int(h * 0.5))
        rx = random.randint(0, max(0, w - rw))
        ry = random.randint(0, max(0, h - rh))
        roi = distorted[ry : ry + rh, rx : rx + rw]
        # Distort ROI
        roi = cv2.resize(roi, (max(1, rw // 4), max(1, rh // 4)), interpolation=cv2.INTER_NEAREST)
        roi = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_NEAREST)
        distorted[ry : ry + rh, rx : rx + rw] = roi

    return distorted


def generate_fake_dataset(num_fakes_per_genuine: int = 2) -> int:
    FAKES_ROOT.mkdir(parents=True, exist_ok=True)
    
    sku_dirs = [d for d in GENUINE_ROOT.iterdir() if d.is_dir()]
    total_generated = 0

    for sku_dir in sku_dirs:
        fake_sku_dir = FAKES_ROOT / sku_dir.name
        fake_sku_dir.mkdir(parents=True, exist_ok=True)

        genuine_images = list(sku_dir.glob("*.jpg")) + list(sku_dir.glob("*.png"))
        if not genuine_images:
            continue

        for img_path in genuine_images:
            try:
                with Image.open(img_path) as img:
                    img_rgb = np.array(img.convert("RGB"))
                
                for idx in range(num_fakes_per_genuine):
                    fake_np = apply_counterfeit_distortions(img_rgb)
                    fake_img = Image.fromarray(fake_np)
                    out_name = f"fake_{idx+1}_{img_path.name}"
                    fake_img.save(fake_sku_dir / out_name)
                    total_generated += 1
            except Exception as e:
                print(f"[warning] Error processing {img_path}: {e}")

    print(f"Generated {total_generated} fake crops across {len(sku_dirs)} SKU directories in {FAKES_ROOT}")
    return total_generated


if __name__ == "__main__":
    generate_fake_dataset(num_fakes_per_genuine=2)
