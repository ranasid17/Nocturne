"""CSV boundaries for the SQLite prediction history migration."""

import csv
import hashlib
from pathlib import Path


EXPORT_FIELDS = (
    "ticker", "timestamp", "date", "direction", "probability_up", "confidence",
    "atr_pct", "volatility_filter_triggered", "volatility_state",
    "volatility_threshold", "model_id", "readiness_status",
)


def read_legacy_csv(source_path):
    source = Path(source_path)
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return source, fingerprint, rows


def write_history_csv(destination, rows):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in EXPORT_FIELDS} for row in rows)
    return len(rows)
