import unittest
from pathlib import Path

from data_loader import load_image_dataset


class DataLoaderTests(unittest.TestCase):
    def test_load_image_dataset_returns_samples_and_labels(self) -> None:
        dataset_root = Path(__file__).resolve().parent / "genuine_split"
        samples, labels, class_names = load_image_dataset(dataset_root, max_samples_per_class=3)

        self.assertGreater(len(samples), 0)
        self.assertEqual(len(samples), len(labels))
        self.assertGreater(len(class_names), 0)
        self.assertTrue(all(label in class_names for label in labels))


if __name__ == "__main__":
    unittest.main()
