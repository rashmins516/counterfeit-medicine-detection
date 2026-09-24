from __future__ import annotations

from typing import Any, Dict


class FrontendViews:
    @staticmethod
    def scan_view(payload: Dict[str, Any]) -> str:
        return f"Scan result: verdict={payload['verdict']}, confidence={payload['confidence']}, record_hash={payload['record_hash']}"

    @staticmethod
    def verify_view(payload: Dict[str, Any]) -> str:
        return f"Verification: {'valid' if payload['verified'] else 'invalid'}"

    @staticmethod
    def trace_view(payload: Dict[str, Any]) -> str:
        sightings = payload.get("sightings", [])
        if not sightings:
            return "No sightings found."
        lines = [f"{idx + 1}. {item['location']} -> {item['verdict']}" for idx, item in enumerate(sightings)]
        return "\n".join(lines)
