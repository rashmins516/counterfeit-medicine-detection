from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from trace_platform.api import TraceAPI
from trace_platform.classifier import BaseClassifier
from trace_platform.ledger import HashChainLedger

app = FastAPI(title="Trace Platform MVP")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

api = TraceAPI(classifier=BaseClassifier(), ledger=HashChainLedger())


class ScanRequest(BaseModel):
    batch_id: str
    location: str
    device_id: str


class TraceRequest(BaseModel):
    batch_id: str


class VerifyRequest(BaseModel):
    record_hash: str


@app.get("/", response_class=HTMLResponse)
def health_check() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.post("/scan")
def scan(request: ScanRequest) -> dict:
    return api.scan(image=None, batch_id=request.batch_id, location=request.location, device_id=request.device_id)


@app.post("/trace")
def trace(request: TraceRequest) -> dict:
    return api.trace(request.batch_id)


@app.post("/verify")
def verify(request: VerifyRequest) -> dict:
    return api.verify(request.record_hash)
