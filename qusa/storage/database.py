"""SQLite connection and schema helpers for local QUSA state."""

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3


SCHEMA_VERSION = 1


def database_path(config, environ=None):
    """Resolve the locally configured database path without creating it."""

    environment = environ if environ is not None else os.environ
    configured = config.get("storage", {}).get("database_path")
    if not configured:
        configured = environment.get("QUSA_DATABASE_PATH")
    if not configured:
        configured = Path(config["data"]["paths"]["predictions_dir"]) / "qusa.sqlite3"
    return Path(configured).expanduser()


def _connect(path):
    connection = sqlite3.connect(path, timeout=5, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    try:
        connection.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        # Some network or read-only filesystems do not support WAL.
        pass
    return connection


@contextmanager
def connect_database(path):
    """Open a configured SQLite database with bounded lock waits."""

    connection = _connect(Path(path))
    try:
        yield connection
    finally:
        connection.close()


def initialize_database(path):
    """Create or upgrade the local database. Safe to call before every use."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect_database(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        applied = {
            row["version"]
            for row in connection.execute("SELECT version FROM schema_migrations")
        }
        if SCHEMA_VERSION not in applied:
            _create_version_one(connection)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) "
                "VALUES (?, datetime('now'))",
                (SCHEMA_VERSION,),
            )
    return path


def schema_version(path):
    """Return the most recently applied schema version, or zero."""

    if not Path(path).exists():
        return 0
    with connect_database(path) as connection:
        row = connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
    return row["version"] or 0


def _create_version_one(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            operation TEXT NOT NULL,
            ticker TEXT,
            status TEXT NOT NULL,
            retry_key TEXT,
            owner_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            error_message TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS unique_retry_runs
            ON runs(operation, ticker, retry_key) WHERE retry_key IS NOT NULL;
        CREATE INDEX IF NOT EXISTS runs_status_created ON runs(status, created_at DESC);

        CREATE TABLE IF NOT EXISTS model_metadata (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id),
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            fingerprint TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS artifacts_run ON artifacts(run_id);

        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
            ticker TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            feature_date TEXT,
            direction TEXT,
            probability_up REAL,
            confidence TEXT,
            atr_pct REAL,
            volatility_filter_triggered INTEGER,
            volatility_state TEXT,
            volatility_threshold REAL,
            readiness_status TEXT,
            readiness_json TEXT NOT NULL DEFAULT '{}',
            model_id TEXT REFERENCES model_metadata(id)
        );
        CREATE INDEX IF NOT EXISTS predictions_ticker_time
            ON predictions(ticker, occurred_at DESC);

        CREATE TABLE IF NOT EXISTS legacy_imports (
            fingerprint TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            imported_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS legacy_rows (
            fingerprint TEXT NOT NULL REFERENCES legacy_imports(fingerprint),
            row_ordinal INTEGER NOT NULL,
            run_id TEXT NOT NULL REFERENCES runs(id),
            PRIMARY KEY (fingerprint, row_ordinal)
        );

        CREATE TABLE IF NOT EXISTS notification_attempts (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id),
            recipients_json TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            claimed_at TEXT,
            completed_at TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS notifications_status_created
            ON notification_attempts(status, created_at);
        """
    )
