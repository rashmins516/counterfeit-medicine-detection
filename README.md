# Counterfeit Medicine Project

Small toolkit and demo for counterfeit medicine image preprocessing, fake-generation, and a minimal traceability MVP (API + ledger).

## Repository layout

- `Medicines.v17i.coco/` — (not committed) COCO-style Roboflow export (train/valid/test).
- `crop_skus_starter.py` — crop boxes from COCO annotations into `genuine_split/`.
- `genuine_split/` — cleaned genuine crops (do not commit large datasets).
- `fakes/` — generated counterfeit/fake variations for testing.
- `trace_system/` — trace platform MVP (FastAPI, ledger, templates, docker-compose).
- `sample/` — tiny sample data for quick demos (safe to commit).

Only source code, documentation, dependency files, and small samples belong in Git.
Local datasets, trained model binaries, uploaded images, runtime ledgers, and generated
outputs are ignored by `.gitignore` and should be supplied or rebuilt locally.

## Quickstart (development)

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the trace API (development mode) and open the frontend:

```bash
cd trace_system
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000/ in a browser to use the demo UI
```

The dashboard accepts a medicine image and writes each result to
`trace_system/data/prediction_ledger.json`. To rebuild the compatible local
authenticity artifact from the generated datasets, run this once from the repo root:

```bash
python train_counterfeit_model.py
```

## Crop dataset (local)

1. Place Roboflow export files under `Medicines.v17i.coco/train`, `.../valid`, `.../test`.
2. Ensure each split contains a `_annotations.coco.json` and images.
3. Run the crop script (creates `genuine_split/`):

```bash
python crop_skus_starter.py
```

Note: Do not commit the full `Medicines.v17i.coco/` dataset to GitHub. Add a small subset to `sample/` if you need example images in the repo.

## Demo ledger and sample data

For a quick demo of the trace system, a minimal sample ledger is available at `trace_system/data/sample_ledger.jsonl` and a tiny sample folder at `sample/`.

## Contributing & license

- See `LICENSE` for terms (MIT).
- Contributions welcome — open a PR with a short description and example data that remains small.

## Contact

If you want me to push these starter files to your GitHub repo, tell me and I can create a branch and open a PR for you.
