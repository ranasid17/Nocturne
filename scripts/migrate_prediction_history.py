"""Import old prediction CSV history or export SQLite history as CSV."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from qusa.storage.runs import RunRepository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="SQLite database path")
    parser.add_argument("--csv", required=True, help="CSV source for import or destination for export")
    parser.add_argument("--dry-run", action="store_true", help="Validate an import without writing")
    parser.add_argument("--export", action="store_true", help="Export SQLite history instead of importing")
    args = parser.parse_args()
    repository = RunRepository(args.database)
    if args.export:
        if args.dry_run:
            parser.error("--dry-run is only available for imports")
        print(f"Exported {repository.export_predictions_csv(args.csv)} prediction(s).")
    else:
        result = repository.import_legacy_csv(args.csv, dry_run=args.dry_run)
        action = "Would import" if args.dry_run else "Imported"
        print(f"{action} {result['rows']} row(s); new rows: {result['imported']}.")


if __name__ == "__main__":
    main()
