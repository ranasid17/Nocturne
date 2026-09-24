---
title: Operations
nav_order: 6
---

# Operations

Nocturne stores run and prediction metadata in SQLite. Raw and processed data,
models, figures, reports, and exported CSVs remain local files. The default
database is `data/predictions/qusa.sqlite3`; set `QUSA_DATABASE_PATH` to use
another location. The server is synchronous, unauthenticated, and intended for
loopback use. Pages hosts documentation only.

## Backup

Stop Flask and any CLI research/import process before a release backup so the
database and artifact folders represent the same point in time. Run from the
repository root. Replace `/path/to/backup` with an existing, private destination.
The SQLite `.backup` command makes a consistent database image even when the
source has a write-ahead log; copying only the `.sqlite3` file does not.

```sh
mkdir -p /path/to/backup
sqlite3 data/predictions/qusa.sqlite3 ".backup /path/to/backup/qusa.sqlite3"
cp -R data saved_models /path/to/backup/
sqlite3 /path/to/backup/qusa.sqlite3 'PRAGMA integrity_check; SELECT * FROM schema_migrations;'
```

The current schema version is **1**. The `schema_migrations` table records
`version` and `applied_at`; the verified fixture reports version 1. If custom
configuration puts artifacts outside `data` or `saved_models`, copy those paths
too. Review `artifacts.path` and `model_metadata.path` in the database to make
sure every referenced file is included:

```sh
sqlite3 /path/to/backup/qusa.sqlite3 'SELECT kind, path FROM artifacts; SELECT path FROM model_metadata;'
```

## Restore and rollback

Stop all app and CLI writers. Restore the database and the matching artifact
folders together to the **same absolute paths** if possible. Do not overwrite a
newer database without first backing it up; otherwise predictions recorded after
the backup will be lost. If restoring to another location, set
`QUSA_DATABASE_PATH` and update custom YAML artifact paths. Stored artifact paths
are absolute in many runs; relocating the files does not rewrite the database.
Verify `PRAGMA integrity_check`, schema version, `GET /api/predictions/history`,
and a known `/api/runs/{run_id}`. Verify each recorded artifact exists before
accepting the restore. Keep the original legacy prediction CSV untouched: the
importer fingerprints its bytes and can rebuild a fresh database from that source.

## Legacy CSV

To rehearse a legacy migration, import into a **copy** of the database. The
`--dry-run` command validates and counts but does not write. Repeat import is
idempotent for the same source fingerprint; a changed source is a new import.
Export writes the canonical prediction columns (including `date`), while
unrecognized legacy columns remain in the imported run's SQLite metadata, not
in the canonical export. Preserve the source CSV for those original values.

```sh
python scripts/migrate_prediction_history.py --database /tmp/qusa-copy.sqlite3 --csv /path/to/legacy.csv --dry-run
python scripts/migrate_prediction_history.py --database /tmp/qusa-copy.sqlite3 --csv /path/to/legacy.csv
python scripts/migrate_prediction_history.py --database /tmp/qusa-copy.sqlite3 --csv /tmp/qusa-export.csv --export
```

The synthetic recovery rehearsal is executable with
`python tests/docs_recovery.py`. It checks backup and restore, schema version,
date round-trip, unknown legacy metadata, source preservation, and artifact
references without touching your live database.
