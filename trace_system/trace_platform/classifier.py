from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from PIL import Image

try:
    from data_loader import preprocess
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data_loader import preprocess

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "live_counterfeit_detector.joblib"


@dataclass
class ClassificationResult:
    verdict: str
    confidence: float
    is_genuine: Optional[bool] = None
    feature_match_score: Optional[float] = None
    predicted_sku: Optional[str] = None


class BaseClassifier:
    """Classifier wrapper that reads the genuine trained counterfeit detector payload.

    The repository already contains a serialized artifact under the project models
    directory. This code uses that artifact as the primary source of truth and
    only falls back to heuristic values when the artifact is missing or broken.
    """

    def __init__(self, model_path: Path | str = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.payload = None
        self.sku_model = None
        self.counterfeit_model = None
        self.sku_classes = []
        self.auth_classes = []
        try:
            if self.model_path.exists():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    payload = joblib.load(self.model_path)
                self.payload = payload
                self.sku_model = payload.get("sku_model")
                self.counterfeit_model = payload.get("counterfeit_model")
                self.sku_classes = list(payload.get("sku_classes", []))
                self.auth_classes = list(payload.get("auth_classes", ["Counterfeit", "Genuine"]))
        except Exception as exc:
            print(f"[warning] Could not load trained model at {self.model_path}: {exc}")
            self.sku_model = None
            self.counterfeit_model = None

    def _image_to_feature_vector(self, image: Any) -> np.ndarray:
        # Use same deterministic RGB preprocessing as the dataset workflow.
        img = preprocess(image, target_size=(32, 32))
        arr = np.asarray(img, dtype=np.float32).reshape(-1)
        return arr.reshape(1, -1)

    def classify(self, image: Any) -> ClassificationResult:
        """Use the trained counterfeit detector artifact when available.

        If the payload is missing, return a stable heuristic response that still
        preserves the API contract for local dev mode.
        """
        if self.counterfeit_model is None:
            return ClassificationResult(verdict="unavailable", confidence=0.0, is_genuine=None)

        try:
            features = self._image_to_feature_vector(image)
            predicted_sku = None
            if self.sku_model is not None and self.sku_classes:
                sku_probs = self.sku_model.predict_proba(features)[0]
                sku_best_idx = int(np.argmax(sku_probs))
                predicted_sku = str(self.sku_classes[sku_best_idx])

            auth_probs = self.counterfeit_model.predict_proba(features)[0]
            auth_idx = int(np.argmax(auth_probs))
            auth_label = str(self.auth_classes[auth_idx]) if self.auth_classes else "Genuine"
            auth_confidence = float(np.max(auth_probs))

            # The serialized payload labels are ordered Counterfeit, Genuine.
            is_genuine = auth_label.lower() == "genuine"
            verdict = "genuine" if is_genuine else "counterfeit"
            return ClassificationResult(
                verdict=verdict,
                confidence=round(auth_confidence, 4),
                is_genuine=is_genuine,
                feature_match_score=round(auth_confidence, 4),
                predicted_sku=predicted_sku,
            )
        except Exception as exc:
            print(f"[warning] Model inference failed: {exc}")
            raise RuntimeError(f"The loaded model cannot process this image: {exc}") from exc


class RuleBasedClassifier(BaseClassifier):
    """A rule-based fallback kept for backward compatibility with API flow."""

    def classify(self, image: Any) -> ClassificationResult:
        return ClassificationResult(verdict="counterfeit", confidence=0.74, is_genuine=False, feature_match_score=0.74)
