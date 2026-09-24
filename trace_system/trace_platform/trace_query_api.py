from __future__ import annotations

from typing import Any, Dict

from .services import ExternalDataService, RouteInference, SpreadPredictor


class TraceQueryAPI:
    def __init__(self, route_inference: RouteInference | None = None, spread_predictor: SpreadPredictor | None = None):
        self.route_inference = route_inference or RouteInference()
        self.spread_predictor = spread_predictor or SpreadPredictor()
        self.external_data = ExternalDataService()

    def get_trace(self, batch_id: str) -> Dict[str, Any]:
        path = self.route_inference.infer_path(batch_id)
        prediction = self.spread_predictor.predict_next_locations(batch_id)
        return {
            "batch_id": batch_id,
            "path": path,
            "predictions": prediction,
            "overlay": self.external_data.get_overlay("default"),
        }
