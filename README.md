# Counterfeit Medicine Project

This workspace contains a starter structure for the counterfeit medicine data pipeline.

## Layout

- Medicines.v17i.coco/: dataset folder for the COCO-style Roboflow export
- crop_skus_starter.py: starter script that crops boxes from COCO annotations into genuine_split
- genuine_split/: cleaned genuine crops will be written here
- fakes/: fake versions will be saved here

## Next steps

1. Place the real Roboflow export files inside Medicines.v17i.coco/train, Medicines.v17i.coco/valid, and Medicines.v17i.coco/test.
2. Ensure each split contains a _annotations.coco.json file and the matching image files.
3. Run:

   python crop_skus_starter.py

4. Review the generated crops in genuine_split and remove poor examples.
