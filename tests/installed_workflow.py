"""Run from outside the checkout after installing the wheel (no test extras)."""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from synthetic_workflow import prepare_fixture


def verify(root, require_base=False):
    import qusa
    from qusa.services import run_model_workflow
    from qusa.storage.runs import RunRepository
    from web_app import create_app
    if require_base:
        assert importlib.util.find_spec("matplotlib") is None
        assert importlib.util.find_spec("ollama") is None
    assert "site-packages" in qusa.__file__, qusa.__file__
    config, config_path = prepare_fixture(root)
    os.environ["QUSA_CONFIG_PATH"] = str(config_path)
    os.environ.pop("QUSA_DATA_ROOT", None)
    os.environ.pop("QUSA_DATABASE_PATH", None)
    with patch("requests.post", side_effect=AssertionError("Unexpected external call")):
        assert run_model_workflow("UPRO", config)["success"]
        client = create_app().test_client()
        for url in ("/", "/api/health", "/static/js/dashboard.js", "/static/css/dashboard.css"):
            assert client.get(url).status_code == 200, url
        response = client.post("/api/predictions/run", json={"ticker": "UPRO"})
        assert response.status_code == 200, response.json
        run_id = response.json["run_id"]
        restarted = create_app().test_client()
        saved = restarted.get("/api/predictions/latest?ticker=UPRO").json["prediction"]
        assert saved["run_id"] == run_id and saved["date"]
        assert restarted.get("/api/predictions/history?ticker=UPRO").json["history"][0]["date"] == saved["date"]
        assert restarted.get(f"/api/runs/{run_id}").json["run"]["status"] == "succeeded"
        model = Path(config["model"]["output"]["model_output_path"]) / "upro_model.pkl"
        backup = model.with_suffix(".backup")
        model.rename(backup)
        try:
            assert restarted.post("/api/predictions/run", json={"ticker": "UPRO"}).status_code == 404
            assert restarted.get("/api/predictions/latest?ticker=UPRO").json["prediction"]["run_id"] == run_id
        finally:
            backup.rename(model)
        features = client.post("/api/pipeline/run", json={"ticker": "FEATURE"})
        assert features.status_code == 200, features.json
    repository = RunRepository.from_config(config)
    exported = root / "export.csv"
    repository.export_predictions_csv(exported)
    with exported.open() as stream:
        assert next(csv.DictReader(stream))["date"] == saved["date"]
    subprocess.run([sys.executable, "-m", "qusa.cli", "--help"], check=True, capture_output=True)
    cli = subprocess.run([sys.executable, "-m", "qusa.cli", "predict", "UPRO"], check=True, capture_output=True, text=True)
    assert json.loads(cli.stdout)["success"]
    print(f"Installed verification passed; browser fixture configuration: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    verify(args.root.resolve(), require_base=args.base_only)
