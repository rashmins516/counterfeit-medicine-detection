from __future__ import annotations

from typing import Any, Dict

from .classifier import BaseClassifier, ClassificationResult
from .ledger import HashChainLedger, LedgerRecord
from .tracer import BatchTracer


class TraceAPI:
    def __init__(self, classifier: BaseClassifier | None = None, ledger: HashChainLedger | None = None):
        self.classifier = classifier or BaseClassifier()
        self.ledger = ledger or HashChainLedger()
        self.tracer = BatchTracer(self.ledger)

    def scan(self, image: Any, batch_id: str, location: str, device_id: str) -> Dict[str, Any]:
        result = self.classifier.classify(image)
        image_hash = "stub-hash"
        if hasattr(image, "filename"):
            image_hash = str(hash(image.filename))
        record = self.ledger.append(
            LedgerRecord(
                batch_id=batch_id,
                location=location,
                device_id=device_id,
                verdict=result.verdict,
                confidence=result.confidence,
                image_hash=image_hash,
            )
        )
        return {
            "verdict": record.verdict,
            "confidence": record.confidence,
            "record_hash": record.record_hash,
            "is_genuine": result.is_genuine,
            "feature_match_score": result.feature_match_score,
            "predicted_sku": result.predicted_sku,
            "trace_id": record.record_hash,
            "geo_node": {"location": location, "device_id": device_id},
        }

    def trace(self, batch_id: str) -> Dict[str, Any]:
        route = self.tracer.infer_route(batch_id)
        return {
            "batch_id": batch_id,
            "sightings": self.tracer.trace(batch_id),
            "route": route,
            "spread_prediction": self.tracer.predict_spread(batch_id),
        }

    def verify(self, record_hash: str) -> Dict[str, Any]:
        return {"record_hash": record_hash, "verified": self.ledger.verify(record_hash)}

    def verify_image(self, image: Any, batch_id: str, location: str, device_id: str) -> Dict[str, Any]:
        return self.scan(image, batch_id, location, device_id)
