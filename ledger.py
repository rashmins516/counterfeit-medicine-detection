from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LEDGER_PATH


@dataclass
class LedgerRecord:
    batch_id: str
    location: str
    device_id: str
    verdict: str
    confidence: float
    image_hash: str
    prev_hash: Optional[str] = None
    record_hash: Optional[str] = None


class HashChainLedger:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or LEDGER_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.path.read_text(encoding="utf-8").strip():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write_records(self, records: List[Dict[str, Any]]) -> None:
        content = "\n".join(json.dumps(record, sort_keys=True) for record in records) + ("\n" if records else "")
        self.path.write_text(content, encoding="utf-8")

    def append(self, record: LedgerRecord) -> LedgerRecord:
        previous_records = self._load_records()
        prev_hash = previous_records[-1]["record_hash"] if previous_records else None
        payload = {
            "batch_id": record.batch_id,
            "location": record.location,
            "device_id": record.device_id,
            "verdict": record.verdict,
            "confidence": record.confidence,
            "image_hash": record.image_hash,
            "prev_hash": prev_hash,
        }
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        record_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        final_record = LedgerRecord(
            batch_id=record.batch_id,
            location=record.location,
            device_id=record.device_id,
            verdict=record.verdict,
            confidence=record.confidence,
            image_hash=record.image_hash,
            prev_hash=prev_hash,
            record_hash=record_hash,
        )
        previous_records.append(asdict(final_record))
        self._write_records(previous_records)
        return final_record

    def verify(self, record_hash: str) -> bool:
        records = self._load_records()
        return any(item.get("record_hash") == record_hash for item in records)

    def get_all(self) -> List[LedgerRecord]:
        records = self._load_records()
        return [LedgerRecord(**record) for record in records]
