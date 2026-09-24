# Traceable Counterfeit Detection MVP

This folder contains the runnable traceability MVP:

1. Classifier
2. Hash-chain ledger
3. Batch tracer
4. API orchestration
5. Frontend shell

## Current scope

- A trained authenticity classifier loaded from `models/live_counterfeit_detector.joblib`
- A hash-chain ledger backed by JSONL for local development
- A prediction ledger with timestamps, filenames, locations, and results
- Batch route tracing with earliest recorded location and counterfeit hotspots
- A FastAPI dashboard with image upload, exports, filters, and live metrics

## Run locally

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
cd trace_system
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. Rebuild the ignored local model from the root with
`python train_counterfeit_model.py` when the dataset changes.
