# Operations

## Local State

QUSA stores run and prediction metadata in SQLite. Raw and processed data, models,
figures, reports, and exported CSVs remain local files. The default database is
`data/predictions/qusa.sqlite3`; set `QUSA_DATABASE_PATH` to use another location.

## Backup

Stop the Flask process before a release backup. Use SQLite's backup command so a
WAL database is captured consistently, then copy the referenced artifact folders.

```bash
sqlite3 data/predictions/qusa.sqlite3 ".backup data/predictions/qusa-backup.sqlite3"
cp -R data saved_models /path/to/backup/
```

Record the schema version with:

```bash
sqlite3 data/predictions/qusa.sqlite3 'SELECT * FROM schema_migrations;'
```

## Restore And Rollback

Stop Flask, restore the database and matching artifact folders together, set
`QUSA_DATABASE_PATH` if the restored database has a different path, and restart
the app. Verify `GET /api/predictions/history` and a known run-detail URL before
accepting the restore. Never overwrite the original legacy prediction CSV: the
importer fingerprints it and can rebuild a fresh database from that preserved source.

To test a migration safely, first run `scripts/migrate_prediction_history.py` with
`--dry-run`, then import into a copy of the database. The importer is idempotent for
the same source fingerprint and row ordinal.
