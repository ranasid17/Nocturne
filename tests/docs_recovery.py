"""Rehearse documented SQLite and legacy CSV recovery with temporary data."""

import csv
from contextlib import closing
import hashlib
from pathlib import Path
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qusa.storage.database import schema_version
from qusa.storage.runs import RunRepository


def main():
    with TemporaryDirectory(prefix="nocturne-docs-recovery-") as directory:
        root = Path(directory)
        source_csv = root / "legacy.csv"
        source_csv.write_text(
            "ticker,timestamp,date,direction,probability_up,legacy_risk\n"
            "UPRO,2025-01-31T20:00:00+00:00,2025-01-31,UP,0.63,unknown-original\n",
            encoding="utf-8",
        )
        source_fingerprint = hashlib.sha256(source_csv.read_bytes()).hexdigest()
        database = root / "live.sqlite3"
        repository = RunRepository(database)
        assert repository.import_legacy_csv(source_csv, dry_run=True)["imported"] == 0
        assert repository.import_legacy_csv(source_csv)["imported"] == 1
        assert repository.import_legacy_csv(source_csv)["imported"] == 0

        artifact = root / "saved_models" / "synthetic-model.pkl"
        artifact.parent.mkdir()
        artifact.write_bytes(b"synthetic fixture only")
        run = repository.create_run("research", ticker="UPRO")
        repository.record_artifact(run["id"], "model", artifact)

        backup_dir = root / "backup"
        backup_dir.mkdir()
        backup_database = backup_dir / "qusa.sqlite3"
        with closing(sqlite3.connect(database)) as live, closing(sqlite3.connect(backup_database)) as backup:
            live.backup(backup)
        shutil.copytree(artifact.parent, backup_dir / "saved_models")

        database.replace(root / "pre-restore.sqlite3")
        shutil.copy2(backup_database, database)
        artifact.unlink()
        shutil.copy2(backup_dir / "saved_models" / artifact.name, artifact)
        restored = RunRepository(database)
        history = restored.list_predictions(ticker="UPRO")
        assert len(history) == 1 and history[0]["date"] == "2025-01-31"
        assert schema_version(database) == 1
        assert (backup_dir / "saved_models" / artifact.name).read_bytes() == artifact.read_bytes()
        assert all(Path(item["path"]).exists() for item in restored.get_run(run["id"])["artifacts"])
        with closing(sqlite3.connect(database)) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 1
        legacy_run = restored.get_run(history[0]["run_id"])
        assert legacy_run["metadata"]["legacy"]["legacy_risk"] == "unknown-original"

        exported = root / "export.csv"
        assert restored.export_predictions_csv(exported) == 1
        with exported.open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        assert row["date"] == "2025-01-31" and row["probability_up"] == "0.63"
        assert hashlib.sha256(source_csv.read_bytes()).hexdigest() == source_fingerprint
        print("Recovery rehearsal passed: schema v1, SQLite backup, artifact paths, idempotent import, dates, legacy metadata, source preservation.")


if __name__ == "__main__":
    main()
