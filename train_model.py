from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_loader import load_counterfeit_dataset, load_image_dataset

ROOT = Path(__file__).resolve().parent
GENUINE_ROOT = ROOT / "genuine_split"
FAKES_ROOT = ROOT / "fakes"
MODELS_ROOT = ROOT / "models"


def extract_visual_features(image: Image.Image, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Extracts a multi-modal feature vector containing:
    1. Spatial RGB pixel intensities (downsampled)
    2. HSV color histogram (capturing ink/dye tones)
    3. LAB color histogram (capturing perceptual color shifts)
    4. Gradient/Texture statistics (Sobel edge energy across channels)
    """
    # Resize for spatial feature
    resized = image.resize(size, Image.Resampling.LANCZOS)
    img_np = np.asarray(resized, dtype=np.float32)
    spatial_feat = img_np.reshape(-1) / 255.0

    # Convert to OpenCV format (uint8 RGB)
    img_cv = np.asarray(image.convert("RGB"), dtype=np.uint8)

    # 1. HSV Histogram
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    # 2. LAB Histogram
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l_hist = cv2.calcHist([lab], [0], None, [16], [0, 256]).flatten()
    a_hist = cv2.calcHist([lab], [1], None, [16], [0, 256]).flatten()
    b_hist = cv2.calcHist([lab], [2], None, [16], [0, 256]).flatten()

    # Normalize histograms
    color_hist = np.concatenate([h_hist, s_hist, v_hist, l_hist, a_hist, b_hist])
    color_hist = color_hist / (color_hist.sum() + 1e-7)

    # 3. Texture / Gradient Energy
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    
    grad_mean = np.mean(mag)
    grad_std = np.std(mag)
    grad_max = np.max(mag)
    grad_hist, _ = np.histogram(mag, bins=16, range=(0, 255))
    grad_hist = grad_hist / (grad_hist.sum() + 1e-7)
    
    texture_feat = np.concatenate([[grad_mean, grad_std, grad_max], grad_hist])

    return np.concatenate([spatial_feat, color_hist, texture_feat])


def extract_features_batch(images: List[Image.Image]) -> np.ndarray:
    return np.stack([extract_visual_features(img) for img in images])


def train_models() -> Dict:
    print("Loading genuine and synthetic counterfeit dataset...")
    images, sku_labels, is_genuine_flags = load_counterfeit_dataset(
        GENUINE_ROOT, FAKES_ROOT, max_samples_per_class=150
    )

    if len(images) < 10:
        raise ValueError("Not enough images found in genuine_split/ to train models.")

    print(f"Extracted {len(images)} samples (Genuine: {sum(is_genuine_flags)}, Counterfeit: {len(is_genuine_flags) - sum(is_genuine_flags)})")
    
    X = extract_features_batch(images)
    y_sku = np.array(sku_labels)
    y_auth = np.array(is_genuine_flags)

    # Encode SKU labels
    sku_encoder = LabelEncoder()
    y_sku_encoded = sku_encoder.fit_transform(y_sku)

    # --- 1. Train SKU Classifier ---
    print("\n--- Training Medicine SKU Classifier ---")
    X_train_sku, X_val_sku, y_train_sku, y_val_sku = train_test_split(
        X, y_sku_encoded, test_size=0.2, random_state=42, stratify=y_sku_encoded
    )
    
    sku_clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    )
    sku_clf.fit(X_train_sku, y_train_sku)
    sku_acc = sku_clf.score(X_val_sku, y_val_sku)
    print(f"SKU Classifier Validation Accuracy: {sku_acc * 100:.2f}%")

    # --- 2. Train Genuine vs. Counterfeit Classifier ---
    print("\n--- Training Genuine vs. Counterfeit Classifier ---")
    X_train_auth, X_val_auth, y_train_auth, y_val_auth = train_test_split(
        X, y_auth, test_size=0.2, random_state=42, stratify=y_auth
    )
    
    auth_clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    )
    auth_clf.fit(X_train_auth, y_train_auth)
    auth_acc = auth_clf.score(X_val_auth, y_val_auth)
    print(f"Counterfeit Detector Validation Accuracy: {auth_acc * 100:.2f}%")
    
    val_preds = auth_clf.predict(X_val_auth)
    print("\nClassification Report (Genuine vs Counterfeit):")
    print(classification_report(y_val_auth, val_preds, target_names=["Counterfeit", "Genuine"]))

    # Save artifacts
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "sku_model": sku_clf,
        "counterfeit_model": auth_clf,
        "sku_classes": sku_encoder.classes_,
        "auth_classes": np.array(["Counterfeit", "Genuine"]),
    }

    model_path = MODELS_ROOT / "counterfeit_detector.joblib"
    joblib.dump(model_payload, model_path)
    print(f"\nSaved trained models to: {model_path}")

    # Backward compatibility artifact
    joblib.dump({"model": sku_clf, "classes": sku_encoder.classes_}, MODELS_ROOT / "baseline_classifier.joblib")
    
    return model_payload


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()
