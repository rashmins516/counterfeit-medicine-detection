from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import preprocess

ROOT = Path(__file__).resolve().parent
GENUINE_ROOT = ROOT / "genuine_split"
FAKES_ROOT = ROOT / "fakes"
MODEL_PATH = ROOT / "models" / "live_counterfeit_detector.joblib"


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def features_for(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        processed = preprocess(image, target_size=(32, 32))
    return np.asarray(processed, dtype=np.float32).reshape(-1)


def main() -> None:
    genuine_files = image_files(GENUINE_ROOT)
    fake_files = image_files(FAKES_ROOT)
    if not genuine_files or not fake_files:
        raise FileNotFoundError("Both genuine_split and fakes must contain image files")

    paths = genuine_files + fake_files
    labels = np.array([1] * len(genuine_files) + [0] * len(fake_files), dtype=np.int8)
    matrix = np.stack([features_for(path) for path in paths])
    x_train, x_test, y_train, y_test = train_test_split(
        matrix, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions, target_names=["Counterfeit", "Genuine"], zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"counterfeit_model": model, "auth_classes": ["Counterfeit", "Genuine"]}, MODEL_PATH)
    MODEL_PATH.with_name("live_counterfeit_detector_metadata.json").write_text(
        json.dumps({"features": "RGB pixels", "target_size": [32, 32], "classes": ["Counterfeit", "Genuine"]}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved compatible authenticity model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
