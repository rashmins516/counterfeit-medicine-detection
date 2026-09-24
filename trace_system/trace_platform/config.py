from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LEDGER_PATH = DATA_DIR / "ledger.jsonl"
GRAPH_STATE_PATH = DATA_DIR / "graph_state.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
