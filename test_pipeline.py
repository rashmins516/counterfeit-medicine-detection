import unittest
from pathlib import Path
import joblib
from PIL import Image
import numpy as np


from data_loader import load_image_dataset, load_counterfeit_dataset
from train_model import extract_visual_features, train_models
from predict import predict_image

ROOT = Path(__file__).resolve().parent
GENUINE_ROOT = ROOT / "genuine_split"
FAKES_ROOT = ROOT / "fakes"


class CounterfeitPipelineTests(unittest.TestCase):
    def test_feature_extraction_returns_correct_shape(self) -> None:
        dummy_img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        features = extract_visual_features(dummy_img, size=(64, 64))
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(features.shape[0], 0)

    def test_load_counterfeit_dataset(self) -> None:
        images, sku_labels, is_genuine_flags = load_counterfeit_dataset(
            GENUINE_ROOT, FAKES_ROOT, max_samples_per_class=2
        )
        self.assertGreater(len(images), 0)
        self.assertEqual(len(images), len(sku_labels))
        self.assertEqual(len(images), len(is_genuine_flags))
        self.assertIn(1, is_genuine_flags)

    def test_model_training_and_prediction(self) -> None:
        model_path = ROOT / "models" / "counterfeit_detector.joblib"
        self.assertTrue(model_path.exists())
        payload = joblib.load(model_path)
        
        self.assertIn("sku_model", payload)
        self.assertIn("counterfeit_model", payload)
        self.assertIn("sku_classes", payload)

        # Test predict_image on a sample genuine crop
        sku_dirs = [d for d in GENUINE_ROOT.iterdir() if d.is_dir()]
        sample_img_files = list(sku_dirs[0].glob("*.jpg"))
        if sample_img_files:
            test_img_path = sample_img_files[0]
            res = predict_image(test_img_path, model_payload=payload)
            self.assertIn("predicted_sku", res)
            self.assertIn("authenticity_status", res)
            self.assertIn("authenticity_confidence_pct", res)
            self.assertTrue(Path(res["annotated_image_path"]).exists())



if __name__ == "__main__":
    unittest.main()
