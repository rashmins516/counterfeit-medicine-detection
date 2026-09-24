from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from .db import Neo4jGraphStore, PostgresLedgerStore


class SightingConsumer:
    """Writes each new sighting into Postgres and Neo4j."""

    def __init__(self, postgres: PostgresLedgerStore | None = None, neo4j: Neo4jGraphStore | None = None):
        self.postgres = postgres or PostgresLedgerStore()
        self.neo4j = neo4j or Neo4jGraphStore()

    def ingest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.postgres.init_schema()
        self.neo4j.init_schema()

        conn = self.postgres.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ledger_records (batch_id, location, device_id, verdict, confidence, image_hash, prev_hash, record_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING record_hash
                    """,
                    (
                        payload["batch_id"],
                        payload["location"],
                        payload["device_id"],
                        payload["verdict"],
                        payload["confidence"],
                        payload["image_hash"],
                        payload.get("prev_hash"),
                        payload["record_hash"],
                    ),
                )
                row = cur.fetchone()
                conn.commit()
                record_hash = row[0] if row else payload["record_hash"]
        finally:
            conn.close()

        with self.neo4j.driver.session() as session:
            session.run(
                """
                MERGE (loc:Location {name: $location})
                MERGE (s:Sighting {id: $record_hash})
                SET s.batch_id = $batch_id,
                    s.verdict = $verdict,
                    s.confidence = $confidence,
                    s.device_id = $device_id,
                    s.image_hash = $image_hash
                MERGE (loc)<-[:SEEN_AT]-(s)
                """,
                {
                    "location": payload["location"],
                    "record_hash": record_hash,
                    "batch_id": payload["batch_id"],
                    "verdict": payload["verdict"],
                    "confidence": payload["confidence"],
                    "device_id": payload["device_id"],
                    "image_hash": payload["image_hash"],
                },
            )

        return {"record_hash": record_hash, "stored": True}


class RouteInference:
    def __init__(self, neo4j: Neo4jGraphStore | None = None):
        self.neo4j = neo4j or Neo4jGraphStore()

    def infer_path(self, batch_id: str) -> List[Dict[str, Any]]:
        with self.neo4j.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Sighting {batch_id: $batch_id})
                RETURN s.location AS location, s.verdict AS verdict
                ORDER BY s.timestamp DESC
                """,
                {"batch_id": batch_id},
            )
            return [dict(record) for record in result]


class SpreadPredictor:
    def __init__(self, neo4j: Neo4jGraphStore | None = None):
        self.neo4j = neo4j or Neo4jGraphStore()

    def predict_next_locations(self, batch_id: str) -> List[Dict[str, Any]]:
        return [{"location": "TBD", "score": 0.0, "reason": "stubbed MVP prediction"}]


class ExternalDataService:
    def get_overlay(self, location: str) -> Dict[str, Any]:
        return {"location": location, "patrol_density": 0.5, "checkpoint_coverage": 0.3, "abuse_density": 0.2}
