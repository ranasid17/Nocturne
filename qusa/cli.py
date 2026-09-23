"""Small installed command-line entry point for the same workflow services."""

import argparse
import json

from qusa.utils.config import load_config
from qusa.utils.errors import safe_error


def main(argv=None):
    parser = argparse.ArgumentParser(description="Nocturne local research workflows")
    parser.add_argument("operation", choices=["features", "predict", "research", "cluster"])
    parser.add_argument("ticker")
    parser.add_argument("--config", help="YAML override (otherwise QUSA_CONFIG_PATH or packaged defaults)")
    parser.add_argument("--fetch", action="store_true", help="Fetch recent provider data for features/predict")
    args = parser.parse_args(argv)
    from qusa.services import make_latest_prediction, run_feature_pipeline, run_model_workflow, run_clustering_workflow
    try:
        if args.operation in {"features", "predict"}:
            workflow = run_feature_pipeline if args.operation == "features" else make_latest_prediction
            result = workflow(args.ticker, fetch_latest=args.fetch, config_path=args.config)
        else:
            workflow = run_model_workflow if args.operation == "research" else run_clustering_workflow
            result = workflow(args.ticker, load_config(args.config))
        print(json.dumps(result, default=str))
        return 0 if result["success"] else 1
    except Exception as exc:
        print(json.dumps({"success": False, "error": safe_error(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
