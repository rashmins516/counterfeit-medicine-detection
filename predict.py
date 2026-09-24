from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import joblib
import numpy as np
from PIL import Image

from train_model import extract_visual_features

ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT / "models"
MODEL_PATH = MODELS_ROOT / "counterfeit_detector.joblib"


def auto_detect_rois(img_np: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect potential medicine box/strip regions of interest using contour analysis."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w, _ = img_np.shape
    min_area = (h * w) * 0.05
    rois = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            rois.append((x, y, x + bw, y + bh))

    # Sort by area descending
    rois.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    
    # Fallback to full image if no ROI found
    if not rois:
        rois = [(0, 0, w, h)]

    return rois


def predict_image(image_path: Path | str, model_payload: Optional[Dict] = None) -> Dict:
    image_file = Path(image_path).resolve()
    if not image_file.exists():
        raise FileNotFoundError(f"Test image not found: {image_file}")

    if model_payload is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run train_model.py first.")
        model_payload = joblib.load(MODEL_PATH)

    sku_model = model_payload["sku_model"]
    counterfeit_model = model_payload.get("counterfeit_model")
    sku_classes = model_payload["sku_classes"]

    with Image.open(image_file) as raw_img:
        img_rgb = np.array(raw_img.convert("RGB"))

    # Extract ROIs or use whole image
    rois = auto_detect_rois(img_rgb)
    x1, y1, x2, y2 = rois[0]  # Take primary ROI
    roi_crop = img_rgb[y1:y2, x1:x2]
    crop_pil = Image.fromarray(roi_crop)

    # Extract visual features
    features = extract_visual_features(crop_pil).reshape(1, -1)

    # Predict SKU
    sku_probs = sku_model.predict_proba(features)[0]
    best_sku_idx = np.argmax(sku_probs)
    predicted_sku = str(sku_classes[best_sku_idx])
    sku_confidence = float(sku_probs[best_sku_idx]) * 100.0

    # Predict Authenticity (Genuine vs Counterfeit)
    if counterfeit_model is not None:
        auth_probs = counterfeit_model.predict_proba(features)[0]
        # Class 1 = Genuine, Class 0 = Counterfeit
        genuine_prob = float(auth_probs[1]) * 100.0
        counterfeit_prob = float(auth_probs[0]) * 100.0
        is_genuine = genuine_prob >= 50.0
        authenticity_label = "GENUINE" if is_genuine else "COUNTERFEIT"
        auth_confidence = genuine_prob if is_genuine else counterfeit_prob
    else:
        authenticity_label = "UNKNOWN (Run train_model.py to enable counterfeit detector)"
        auth_confidence = 0.0

    result = {
        "image_file": str(image_file),
        "predicted_sku": predicted_sku,
        "sku_confidence_pct": round(sku_confidence, 2),
        "authenticity_status": authenticity_label,
        "authenticity_confidence_pct": round(auth_confidence, 2),
        "roi_box": [x1, y1, x2, y2],
    }

    # Generate annotated image
    annotated = img_rgb.copy()
    color = (0, 255, 0) if authenticity_label == "GENUINE" else (255, 0, 0)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    label_text = f"{predicted_sku} | {authenticity_label} ({result['authenticity_confidence_pct']}%)"
    cv2.putText(
        annotated, label_text, (max(10, x1), max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )

    out_img_path = ROOT / "prediction_result.jpg"
    Image.fromarray(annotated).save(out_img_path)
    result["annotated_image_path"] = str(out_img_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfeit Medicine Detector")
    parser.add_argument("--image", type=str, help="Path to test image file")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        # Pick default image from test directory if available
        test_dir = ROOT / "test"
        if test_dir.exists():
            test_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if test_files:
                image_path = str(test_files[0])

    if not image_path:
        print("Please provide a test image path: python predict.py --image <path_to_image>")
        return

    print(f"\nAnalyzing test image: {image_path}...")
    res = predict_image(image_path)

    print("\n================ DETECTOR ANALYSIS REPORT ================")
    print(f"File Path           : {res['image_file']}")
    print(f"Predicted Medicine  : {res['predicted_sku']} ({res['sku_confidence_pct']}% confidence)")
    print(f"Authenticity Status : {res['authenticity_status']} ({res['authenticity_confidence_pct']}% confidence)")
    print(f"Annotated Image Saved: {res['annotated_image_path']}")
    print("==========================================================")


if __name__ == "__main__":
    main()
