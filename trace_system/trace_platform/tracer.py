from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .ledger import HashChainLedger, LedgerRecord


class BatchTracer:
    """Reads the ledger and reconstructs ordered sightings for a batch.

    It also exposes route inference and hotspot prediction helpers that can later
    be backed by PostgreSQL and Neo4j graph stores.
    """

    def __init__(self, ledger: HashChainLedger | None = None):
        self.ledger = ledger or HashChainLedger()

    def trace(self, batch_id: str) -> List[Dict[str, object]]:
        records = [record for record in self.ledger.get_all() if record.batch_id == batch_id]
        records.sort(key=lambda item: item.record_hash or "")
        return [
            {
                "location": record.location,
                "device_id": record.device_id,
                "verdict": record.verdict,
                "confidence": record.confidence,
                "record_hash": record.record_hash,
            }
            for record in records
        ]

    def infer_route(self, batch_id: str) -> Dict[str, object]:
        """Return a deterministic route summary from the ordered sightings."""
        records = [record for record in self.ledger.get_all() if record.batch_id == batch_id]
        locations = [record.location for record in records]
        devices = [record.device_id for record in records]
        counter = Counter(locations)
        return {
            "batch_id": batch_id,
            "locations": locations,
            "devices": devices,
            "anomaly_score": round(min(1.0, max(0.0, (len(records) / max(1, len(locations)) * 0.25))), 4),
            "hotspots": [location for location, count in counter.items() if count > 1],
        }

    def predict_spread(self, batch_id: str) -> Dict[str, object]:
        """Return a lightweight route-risk/spread heuristic based on location count and confidence."""
        records = [record for record in self.ledger.get_all() if record.batch_id == batch_id]
        if not records:
            return {"risk_level": "unknown", "score": 0.0, "locations": []}

        locations = sorted({record.location for record in records})
        negatives = sum(1 for record in records if record.verdict.lower() in {"fake", "suspicious"})
        max_confidence = max([record.confidence for record in records], default=0.0)
        score = min(1.0, (negatives / max(1, len(records))) + (max_confidence / 2.0))

        if score >= 0.7:
            risk = "high"
        elif score >= 0.35:
            risk = "medium"
        else:
            risk = "low"

        return {"risk_level": risk, "score": round(score, 4), "locations": locations}
