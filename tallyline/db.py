"""SQLite tables and helpers. One file: data/tallyline.db."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .settings import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY, case_type TEXT NOT NULL, policy_number TEXT, policy_start TEXT,
    incident_date TEXT, amount REAL, currency TEXT, provider TEXT, status TEXT NOT NULL DEFAULT 'Open',
    risk_score REAL, risk_level TEXT, main_issue TEXT, is_sample INTEGER NOT NULL DEFAULT 0,
    features_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS past_cases (
    id TEXT PRIMARY KEY, case_type TEXT NOT NULL, incident_date TEXT, provider TEXT, amount REAL,
    currency TEXT, outcome TEXT, summary TEXT, note TEXT, is_sample INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS analyses (
    case_id TEXT PRIMARY KEY, result_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, decision TEXT NOT NULL,
    comment TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS asks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, question TEXT NOT NULL,
    answer_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS passages (
    id TEXT PRIMARY KEY, index_name TEXT NOT NULL, text TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS indexed_files (
    key TEXT PRIMARY KEY, index_name TEXT NOT NULL, content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS citation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, context TEXT NOT NULL,
    sentence TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


def now() -> str:
    """Current UTC time as an ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(settings: Settings) -> sqlite3.Connection:
    """Open the database, creating the tables if needed."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def query(settings: Settings, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as dictionaries."""
    conn = connect(settings)
    try:
        return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]
    finally:
        conn.close()


def execute(settings: Settings, sql: str, params: Iterable[Any] = ()) -> None:
    """Run one write statement."""
    conn = connect(settings)
    try:
        with conn:
            conn.execute(sql, tuple(params))
    finally:
        conn.close()


def sync_cases(settings: Settings) -> int:
    """Load data/cases/*/case.json into the cases table, keeping status and risk values."""
    count = 0
    conn = connect(settings)
    try:
        with conn:
            seen = []
            for path in sorted(settings.cases_dir.glob("*/case.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                seen.append(record["case_id"])
                conn.execute(
                    """INSERT INTO cases (id, case_type, policy_number, policy_start, incident_date, amount,
                           currency, provider, is_sample, features_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET case_type=excluded.case_type,
                           policy_number=excluded.policy_number, policy_start=excluded.policy_start,
                           incident_date=excluded.incident_date, amount=excluded.amount,
                           currency=excluded.currency, provider=excluded.provider,
                           is_sample=excluded.is_sample, features_json=excluded.features_json""",
                    (record["case_id"], record["case_type"], record.get("policy_number"),
                     record.get("policy_start"), record.get("incident_date"), record.get("amount"),
                     record.get("currency", ""), record.get("provider"), int(record.get("sample", False)),
                     json.dumps(record.get("features", {}))),
                )
                count += 1
            placeholders = ",".join("?" * len(seen))
            conn.execute(f"DELETE FROM cases WHERE id NOT IN ({placeholders})", seen)
    finally:
        conn.close()
    return count


def sync_past_cases(settings: Settings) -> int:
    """Replace the past_cases table with the contents of past_cases.jsonl."""
    if not settings.past_cases_path.exists():
        return 0
    rows = read_jsonl(settings.past_cases_path)
    conn = connect(settings)
    try:
        with conn:
            conn.execute("DELETE FROM past_cases")
            conn.executemany(
                """INSERT INTO past_cases (id, case_type, incident_date, provider, amount, currency,
                       outcome, summary, note, is_sample) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(r["past_case_id"], r["case_type"], r.get("incident_date"), r.get("provider"),
                  r.get("amount"), r.get("currency", ""), r["outcome"], r["summary"], r.get("note", ""),
                  int(r.get("sample", False))) for r in rows],
            )
    finally:
        conn.close()
    return len(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def log_removal(settings: Settings, case_id: str | None, context: str, sentence: str, reason: str) -> None:
    """Record a sentence removed by the citation check."""
    execute(
        settings,
        "INSERT INTO citation_log (case_id, context, sentence, reason, created_at) VALUES (?, ?, ?, ?, ?)",
        (case_id, context, sentence, reason, now()),
    )
