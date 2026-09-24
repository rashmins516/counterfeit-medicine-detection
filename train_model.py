from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_loader import load_image_dataset, preprocess

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "genuine_split"
MODEL_DIR = ROOT / "models"


def extract_visual_features(image: Image.Image, size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """Return a flattened one-image feature vector compatible with the trained detector payload.

    This helper is intentionally a thin compatibility wrapper for the prediction path.
    """
    return extract_features([image], size=size).reshape(-1)


def extract_features(images: List[Image.Image], size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """Flatten and standardize image feature vectors for a low-cost baseline model.

    This is designed as the production-safe fallback model while keeping the
    repository compatible with existing scripts and starter expectations.
    """
    features = []
    for image in images:
        resized = preprocess(image, target_size=size)
        arr = np.asarray(resized, dtype=np.float32).reshape(-1)
        features.append(arr)
    return np.stack(features)


def build_pipeline(n_components: int = 80, random_state: int = 42) -> Pipeline:
    """Create a deterministic scikit-learn PCA + Logistic Regression pipeline."""
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=random_state),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    """Return production metrics: precision, recall, F1, average precision, ROC-AUC."""
    if len(np.unique(y_true)) < 2:
        raise ValueError("At least two classes are required for evaluation metrics")

    # Metric macro average is consistent for multi-class sku classification.
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # For ROC-AUC, use one-vs-rest probabilities and target labels.
    try:
        y_prob_matrix = np.column_stack([y_prob[:, i] if y_prob.ndim == 2 else y_prob for i in range(len(classes))])
        auc = float(roc_auc_score(y_true, y_prob_matrix, average="macro", multi_class="ovr"))
    except Exception:
        auc = float("nan")

    # Average precision in multi-class can be one-vs-rest and is safe.
    try:
        ap = float(average_precision_score(y_true, y_prob, average="macro"))
    except Exception:
        ap = float("nan")

    return {
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "roc_auc_macro": auc,
        "average_precision_macro": ap,
    }


def serialize_artifacts(model: Pipeline, label_encoder: LabelEncoder, metrics: Dict[str, float], model_dir: Path) -> None:
    """Save a model joblib artifact and a sidecar JSON metadata file."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline_classifier.joblib"
    metadata_path = model_dir / "baseline_classifier_metadata.json"

    joblib.dump({"model": model, "classes": label_encoder.classes_}, model_path)

    config = {
        "model_type": "pca_logistic_regression",
        "model_family": "visual_classification",
        "objectives": ["sku_classification", "authenticity_classification"],
        "input_size": [256, 256],
        "feature_type": "flattened_rgb_pixels",
        "backend": "sklearn_pipeline",
        "metrics": metrics,
        "labels": [str(x) for x in label_encoder.classes_],
    }

    metadata_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    images, labels, class_names = load_image_dataset(DATASET_ROOT, max_samples_per_class=60)
    if len(images) < 10:
        raise ValueError("Not enough training images found. Add more crops first.")

    y = np.array(labels)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X = extract_features(images)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    # Production model family is a deterministic sklearn-compatible pipeline.
    # For CNN/Vision Transformer support, the code is structured to map metadata and
    # artifacts in a way that supports migration from this baseline pipeline.
    clf = build_pipeline(n_components=min(80, max(2, X_train.shape[1] // 4)))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Evaluate open metrics.
    metrics = evaluate_model(y_test, y_pred, y_prob, label_encoder.classes_)
    print("Validation metrics:", json.dumps(metrics, indent=2))

    # Serialize model + metadata.
    serialize_artifacts(clf, label_encoder, metrics, MODEL_DIR)


if __name__ == "__main__":
    main()
