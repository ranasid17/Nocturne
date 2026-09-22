"""Repository for durable local run, prediction, and notification state."""

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from .database import connect_database, database_path, initialize_database


RUN_TRANSITIONS = {
    "queued": {"running", "failed", "interrupted"},
    "running": {"succeeded", "failed", "interrupted"},
    "succeeded": set(),
    "failed": set(),
    "interrupted": set(),
}


class RunStateError(ValueError):
    """Raised for an invalid or unauthorized run transition."""


class DuplicateRunError(ValueError):
    """Raised when a retry key already identifies an existing run."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value or {}, default=str, sort_keys=True)


def _parse(value):
    return json.loads(value or "{}")


def _safe_error(error):
    return str(error).replace("\n", " ")[:500]


class RunRepository:
    """Small SQLite repository with deliberately short write transactions."""

    def __init__(self, path):
        self.path = initialize_database(path)

    @classmethod
    def from_config(cls, config):
        return cls(database_path(config))

    def create_run(self, operation, ticker=None, retry_key=None, owner_id=None, metadata=None):
        run_id = str(uuid4())
        now = _now()
        try:
            with connect_database(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        "INSERT INTO runs(id, operation, ticker, status, retry_key, owner_id, "
                        "metadata_json, created_at) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)",
                        (run_id, operation, ticker, retry_key, owner_id, _json(metadata), now),
                    )
                    connection.execute("COMMIT")
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
        except sqlite3.IntegrityError as exc:
            raise DuplicateRunError("A run with this retry key already exists.") from exc
        return self.get_run(run_id)

    def transition_run(self, run_id, status, owner_id=None, error_message=None):
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
                if row is None:
                    raise KeyError(run_id)
                if status not in RUN_TRANSITIONS.get(row["status"], set()):
                    raise RunStateError(f"Cannot transition {row['status']} run to {status}.")
                if row["owner_id"] and owner_id != row["owner_id"]:
                    raise RunStateError("Run ownership does not match.")
                started_at = _now() if status == "running" else row["started_at"]
                completed_at = _now() if status in {"succeeded", "failed", "interrupted"} else None
                connection.execute(
                    "UPDATE runs SET status = ?, started_at = ?, completed_at = ?, error_message = ? "
                    "WHERE id = ?",
                    (status, started_at, completed_at, _safe_error(error_message) if error_message else None, run_id),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get_run(run_id)

    def recover_abandoned_runs(self):
        with connect_database(self.path) as connection:
            cursor = connection.execute(
                "UPDATE runs SET status = 'interrupted', completed_at = ?, "
                "error_message = 'Recovered after an interrupted local process.' "
                "WHERE status IN ('queued', 'running')",
                (_now(),),
            )
        return cursor.rowcount

    def record_model(self, model_id, path, metadata=None):
        with connect_database(self.path) as connection:
            connection.execute(
                "INSERT INTO model_metadata(id, path, metadata_json, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET path = excluded.path, metadata_json = excluded.metadata_json",
                (str(model_id), str(path), _json(metadata), _now()),
            )

    def record_artifact(self, run_id, kind, path, fingerprint=None):
        with connect_database(self.path) as connection:
            connection.execute(
                "INSERT INTO artifacts(id, run_id, kind, path, fingerprint, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), run_id, kind, str(path), fingerprint, _now()),
            )

    def record_prediction(self, run_id, log_entry):
        readiness = log_entry.get("readiness") or {}
        with connect_database(self.path) as connection:
            connection.execute(
                "INSERT INTO predictions(id, run_id, ticker, occurred_at, feature_date, direction, "
                "probability_up, confidence, atr_pct, volatility_filter_triggered, volatility_state, "
                "volatility_threshold, readiness_status, readiness_json, model_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()), run_id, str(log_entry.get("ticker", "")).upper(),
                    str(log_entry.get("timestamp", _now())), str(log_entry.get("date")) if log_entry.get("date") is not None else None,
                    log_entry.get("direction"), _number(log_entry.get("probability_up")) if log_entry.get("probability_up") is not None else None, log_entry.get("confidence"),
                    _number(log_entry.get("atr_pct")) if log_entry.get("atr_pct") is not None else None, int(bool(log_entry.get("volatility_filter_triggered"))),
                    log_entry.get("volatility_state"), _number(log_entry.get("volatility_threshold")) if log_entry.get("volatility_threshold") is not None else None,
                    log_entry.get("readiness_status"), _json(readiness), log_entry.get("model_id"),
                ),
            )

    def get_run(self, run_id):
        with connect_database(self.path) as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            artifacts = connection.execute("SELECT kind, path, fingerprint FROM artifacts WHERE run_id = ?", (run_id,)).fetchall()
        result = dict(row)
        result["metadata"] = _parse(result.pop("metadata_json"))
        result["artifacts"] = [dict(item) for item in artifacts]
        return result

    def list_predictions(self, ticker=None, limit=50, offset=0):
        clauses, values = [], []
        if ticker:
            clauses.append("p.ticker = ?")
            values.append(ticker.upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT p.*, r.id AS run_id FROM predictions p JOIN runs r ON r.id = p.run_id "
            f"{where} ORDER BY p.occurred_at DESC, p.id DESC LIMIT ? OFFSET ?"
        )
        with connect_database(self.path) as connection:
            rows = connection.execute(query, (*values, limit, offset)).fetchall()
        return [self._prediction_row(row) for row in rows]

    def latest_prediction(self, ticker=None):
        rows = self.list_predictions(ticker=ticker, limit=1)
        return rows[0] if rows else None

    def _prediction_row(self, row):
        result = dict(row)
        result["timestamp"] = result.pop("occurred_at")
        result["readiness"] = _parse(result.pop("readiness_json"))
        result["volatility_filter_triggered"] = bool(result["volatility_filter_triggered"])
        result.pop("id", None)
        return result

    def import_legacy_csv(self, source_path, dry_run=False):
        source = Path(source_path)
        payload = source.read_bytes()
        fingerprint = hashlib.sha256(payload).hexdigest()
        with source.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if dry_run:
            return {"dry_run": True, "fingerprint": fingerprint, "rows": len(rows), "imported": 0}
        with connect_database(self.path) as connection:
            if connection.execute("SELECT 1 FROM legacy_imports WHERE fingerprint = ?", (fingerprint,)).fetchone():
                return {"dry_run": False, "fingerprint": fingerprint, "rows": len(rows), "imported": 0}
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("INSERT INTO legacy_imports VALUES (?, ?, ?)", (fingerprint, str(source), _now()))
                for ordinal, row in enumerate(rows, start=1):
                    run_id = f"legacy-{fingerprint[:16]}-{ordinal}"
                    timestamp = row.get("timestamp") or _now()
                    ticker = (row.get("ticker") or "UNKNOWN").upper()
                    connection.execute(
                        "INSERT INTO runs(id, operation, ticker, status, metadata_json, created_at, started_at, completed_at) "
                        "VALUES (?, 'legacy_import', ?, 'succeeded', ?, ?, ?, ?)",
                        (run_id, ticker, _json({"legacy": row}), timestamp, timestamp, timestamp),
                    )
                    connection.execute("INSERT INTO legacy_rows VALUES (?, ?, ?)", (fingerprint, ordinal, run_id))
                    connection.execute(
                        "INSERT INTO predictions(id, run_id, ticker, occurred_at, feature_date, direction, probability_up, confidence, atr_pct, volatility_filter_triggered, volatility_state, volatility_threshold, readiness_status, readiness_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')",
                        (str(uuid4()), run_id, ticker, timestamp, row.get("date"), row.get("direction"),
                         _number(row.get("probability_up")), row.get("confidence"), _number(row.get("atr_pct")),
                         _truthy(row.get("volatility_filter_triggered")), row.get("volatility_state"),
                         _number(row.get("volatility_threshold")), row.get("readiness_status")),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return {"dry_run": False, "fingerprint": fingerprint, "rows": len(rows), "imported": len(rows)}

    def export_predictions_csv(self, destination):
        rows = self.list_predictions(limit=1_000_000)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = ["ticker", "timestamp", "date", "direction", "probability_up", "confidence", "atr_pct", "volatility_filter_triggered", "volatility_state", "volatility_threshold", "model_id", "readiness_status"]
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{key: row.get(key) for key in fields} for row in rows])
        return len(rows)

    def enqueue_notification(self, run_id, recipients, payload):
        notification_id = str(uuid4())
        with connect_database(self.path) as connection:
            connection.execute(
                "INSERT INTO notification_attempts(id, run_id, recipients_json, payload_json, status, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?)",
                (notification_id, run_id, _json(recipients), _json(payload), _now()),
            )
        return notification_id

    def claim_notification(self):
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM notification_attempts WHERE status IN ('pending', 'failed') "
                    "ORDER BY created_at LIMIT 1"
                ).fetchone()
                if row is None:
                    connection.execute("COMMIT")
                    return None
                connection.execute(
                    "UPDATE notification_attempts SET status = 'sending', attempts = attempts + 1, claimed_at = ? WHERE id = ?",
                    (_now(), row["id"]),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = dict(row)
        result["recipients"] = _parse(result.pop("recipients_json"))
        result["payload"] = _parse(result.pop("payload_json"))
        return result

    def complete_notification(self, notification_id, status, error_message=None):
        if status not in {"sent", "failed", "unknown"}:
            raise ValueError("Notification status must be sent, failed, or unknown.")
        with connect_database(self.path) as connection:
            connection.execute(
                "UPDATE notification_attempts SET status = ?, completed_at = ?, error_message = ? WHERE id = ?",
                (status, _now(), _safe_error(error_message) if error_message else None, notification_id),
            )

    def recover_sending_notifications(self):
        with connect_database(self.path) as connection:
            cursor = connection.execute(
                "UPDATE notification_attempts SET status = 'unknown', completed_at = ?, "
                "error_message = 'Delivery outcome unknown after worker interruption.' WHERE status = 'sending'",
                (_now(),),
            )
        return cursor.rowcount


def _number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _truthy(value):
    return int(str(value).strip().lower() in {"1", "true", "yes"})
