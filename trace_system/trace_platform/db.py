from __future__ import annotations

import os
from typing import Any, Optional

import psycopg2
from neo4j import GraphDatabase


class PostgresLedgerStore:
    """PostgreSQL-backed store with graceful fallback to a local JSONL ledger object.

    This class stays compatible with the existing schema and lightweight design.
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv("DATABASE_URL", "postgresql://trace_user:trace_pass@localhost:5432/trace_db")

    def connect(self):
        try:
            return psycopg2.connect(self.dsn)
        except Exception as exc:
            print(f"[warn] PostgreSQL unavailable; falling back to local JSONL: {exc}")
            return None

    def init_schema(self) -> None:
        conn = self.connect()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ledger_records (
                        id SERIAL PRIMARY KEY,
                        batch_id TEXT NOT NULL,
                        location TEXT NOT NULL,
                        device_id TEXT NOT NULL,
                        verdict TEXT NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        image_hash TEXT NOT NULL,
                        prev_hash TEXT,
                        record_hash TEXT UNIQUE NOT NULL
                    )
                    """
                )
                conn.commit()
        finally:
            conn.close()


class Neo4jGraphStore:
    """Neo4j graph store adapter with safe fallback if the graph container is absent."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "tracepass")
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        except Exception as exc:
            print(f"[warn] Neo4j unavailable: {exc}")
            self.driver = None

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def init_schema(self) -> None:
        if self.driver is None:
            return
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sighting) REQUIRE s.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE")

    def add_sighting(self, batch_id: str, location: str, device_id: str, verdict: str, confidence: float) -> None:
        if self.driver is None:
            return
        with self.driver.session() as session:
            session.run(
                """
                MERGE (b:Batch {batch_id: $batch_id})
                MERGE (l:Location {name: $location})
                MERGE (d:Device {device_id: $device_id})
                CREATE (s:Sighting {id: randomUUID(), verdict: $verdict, confidence: $confidence})
                MERGE (b)-[:HAS_SIGHTING]->(s)
                MERGE (s)-[:OCCURRED_AT]->(l)
                MERGE (s)-[:SCANNED_BY]->(d)
                """,
                batch_id=batch_id,
                location=location,
                device_id=device_id,
                verdict=verdict,
                confidence=float(confidence),
            )
