import csv
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from qusa.storage.database import connect_database
from qusa.storage.runs import RunRepository, RunStateError
from qusa.utils.settings import load_settings
from synthetic_workflow import prepare_fixture


def test_packaged_defaults_and_override_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_settings(environ={})
    assert config["data"]["paths"]["raw_data_dir"] == str(tmp_path / "data/raw")
    assert config["model"]["output"]["model_output_path"] == str(tmp_path / "saved_models")
    first, second = tmp_path / "one.yaml", tmp_path / "two.yaml"
    first.write_text("value: explicit\n")
    second.write_text("value: environment\n")
    assert load_settings(first, {"QUSA_CONFIG_PATH": str(second)})["value"] == "explicit"
    assert load_settings(environ={"QUSA_CONFIG_PATH": str(second)})["value"] == "environment"


def test_optional_plotting_is_not_imported():
    code = """
import sys
class BlockPlots:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('matplotlib', 'ollama')):
            raise AssertionError('Optional library imported: ' + fullname)
sys.meta_path.insert(0, BlockPlots())
import qusa.services
from web_app import create_app
assert create_app().test_client().get('/').status_code == 200
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_prediction_commit_rolls_back_and_hides_failed_results(tmp_path, monkeypatch):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    successful = repository.create_run("prediction", "UPRO")
    repository.transition_run(successful["id"], "running")
    entry = {"ticker": "UPRO", "date": "2026-03-06", "timestamp": "2026-03-06"}
    repository.complete_prediction(successful["id"], entry)
    failed = repository.create_run("prediction", "UPRO")
    repository.transition_run(failed["id"], "running")
    insert = repository._insert_prediction

    def fail_after_insert(*args):
        insert(*args)
        raise OSError("SENTINEL_SECRET")

    monkeypatch.setattr(repository, "_insert_prediction", fail_after_insert)
    with pytest.raises(OSError):
        repository.complete_prediction(failed["id"], entry)
    with connect_database(repository.path) as db:
        assert db.execute("SELECT count(*) FROM predictions WHERE run_id = ?", (failed["id"],)).fetchone()[0] == 0
    repository.transition_run(failed["id"], "failed", error_message=OSError("SENTINEL_SECRET"))
    assert "SENTINEL_SECRET" not in json.dumps(repository.get_run(failed["id"]))
    # Even rows produced by older, non-atomic clients must not be visible.
    monkeypatch.setattr(repository, "_insert_prediction", insert)
    repository.record_prediction(failed["id"], entry)
    restarted = RunRepository(repository.path)
    assert restarted.latest_prediction()["run_id"] == successful["id"]
    assert len(restarted.list_predictions()) == 1
    with pytest.raises(RunStateError):
        restarted.complete_prediction(failed["id"], entry)


def test_owner_validation_and_date_roundtrip(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    run = repository.create_run("prediction", "UPRO", owner_id="owner")
    repository.transition_run(run["id"], "running", owner_id="owner")
    with pytest.raises(RunStateError):
        repository.complete_prediction(run["id"], {"ticker": "UPRO"})
    repository.complete_prediction(run["id"], {"ticker": "UPRO", "date": "2026-03-06"}, owner_id="owner")
    exported = tmp_path / "export.csv"
    repository.export_predictions_csv(exported)
    with exported.open() as stream:
        assert next(csv.DictReader(stream))["date"] == "2026-03-06"
    imported = RunRepository(tmp_path / "import.sqlite3")
    imported.import_legacy_csv(exported)
    assert imported.latest_prediction()["date"] == "2026-03-06"
    assert imported.latest_prediction()["volatility_state"] == "unknown"
    assert imported.latest_prediction()["volatility_filter_triggered"] is None


def test_real_flask_prediction_restart_and_failure(tmp_path, monkeypatch, caplog):
    from qusa.services.research_service import run_model_workflow
    from web_app import create_app
    config, path = prepare_fixture(tmp_path)
    monkeypatch.setenv("QUSA_CONFIG_PATH", str(path))
    monkeypatch.delenv("QUSA_DATABASE_PATH", raising=False)
    monkeypatch.delenv("QUSA_DATA_ROOT", raising=False)
    result = run_model_workflow("UPRO", config)
    assert result["success"]
    client = create_app().test_client()
    prediction = client.post("/api/predictions/run", json={"ticker": "UPRO"})
    assert prediction.status_code == 200
    run_id = prediction.json["run_id"]
    restarted = create_app().test_client()
    saved = restarted.get("/api/predictions/latest?ticker=UPRO").json["prediction"]
    assert saved["run_id"] == run_id
    assert saved["date"] == saved["feature_date"]
    assert saved["freshness"]["status"] == "stale"
    assert restarted.get("/api/predictions/history?ticker=UPRO").json["history"][0]["date"]
    assert restarted.get(f"/api/runs/{run_id}").json["run"]["status"] == "succeeded"
    monkeypatch.setattr("qusa.services.prediction_service.make_prediction", Mock(side_effect=RuntimeError("Bearer SENTINEL_SECRET")))
    assert client.post("/api/predictions/run", json={"ticker": "UPRO"}).status_code == 500
    assert client.get("/api/predictions/latest?ticker=UPRO").json["prediction"]["run_id"] == run_id
    assert "SENTINEL_SECRET" not in caplog.text
    repo = RunRepository.from_config(config)
    with connect_database(repo.path) as db:
        rows = db.execute("SELECT * FROM runs WHERE status='failed'").fetchall()
    assert len(rows) == 1
    assert "SENTINEL_SECRET" not in client.get(f"/api/runs/{rows[0]['id']}").text


def test_environment_selected_feature_pipeline(tmp_path, monkeypatch):
    from qusa.services import run_feature_pipeline
    config, path = prepare_fixture(tmp_path)
    monkeypatch.setenv("QUSA_CONFIG_PATH", str(path))
    result = run_feature_pipeline("FEATURE")
    assert result["success"]
    assert Path(result["output_path"]).parent == Path(config["data"]["paths"]["processed_data_dir"])


def test_optional_reports_are_isolated_and_artifacts_are_run_owned(tmp_path, monkeypatch):
    from qusa.services.research_service import run_model_workflow
    config, _ = prepare_fixture(tmp_path)
    config["backtest"]["save_plots"] = True
    calls = Mock(side_effect=AssertionError("SENTINEL_SECRET"))
    monkeypatch.setattr("requests.post", calls)
    first = run_model_workflow("UPRO", config)
    assert first["success"]
    calls.assert_not_called()
    config["reporting"].update(enabled=True, enabled_report_types=["training", "evaluation", "backtest", "interpretation"])
    second = run_model_workflow("UPRO", config)
    assert second["success"]
    assert second["reports"]["training"]["status"] == "failed"
    assert second["reports"]["interpretation"]["status"] == "succeeded"
    assert "SENTINEL_SECRET" not in json.dumps(second)
    repository = RunRepository.from_config(config)
    artifact_sets = []
    for result in (first, second):
        run = repository.get_run(result["run_id"])
        assert run["metadata"]["phases"]["backtest"]["status"] == "succeeded"
        artifacts = {item["kind"]: Path(item["path"]) for item in run["artifacts"]}
        assert {"model", "training_metrics", "evaluation_metrics", "backtest_metrics", "backtest_results", "backtest_plot"} <= artifacts.keys()
        assert all(path.exists() and result["run_id"] in str(path) for path in artifacts.values())
        artifact_sets.append(set(artifacts.values()))
    assert artifact_sets[0].isdisjoint(artifact_sets[1])


def test_disabled_outputs_and_cli_parity(tmp_path, monkeypatch, capsys):
    from qusa.cli import main
    from qusa.services.research_service import run_model_workflow
    import yaml
    config, path = prepare_fixture(tmp_path)
    config["training"] = {"save_results": False}
    config["evaluation"]["save_results"] = False
    config["backtest"]["save_results"] = False
    path.write_text(yaml.safe_dump(config))
    direct = run_model_workflow("UPRO", config)
    monkeypatch.setenv("QUSA_CONFIG_PATH", str(path))
    assert main(["research", "UPRO"]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert {name: phase["status"] for name, phase in cli["phases"].items()} == {
        name: phase["status"] for name, phase in direct["phases"].items()
    }
    assert cli["phases"]["evaluation"]["metrics"]["accuracy"] == direct["phases"]["evaluation"]["metrics"]["accuracy"]
    assert cli["phases"]["backtest"]["metrics"]["strategy_return"] == direct["phases"]["backtest"]["metrics"]["strategy_return"]
    repo = RunRepository.from_config(config)
    assert {a["kind"] for a in repo.get_run(cli["run_id"])["artifacts"]} == {"model"}


def test_clustering_preserves_two_runs_for_each_ticker(tmp_path):
    from qusa.services.clustering_service import run_clustering_workflow
    config, _ = prepare_fixture(tmp_path)
    config["clustering"]["save_plots"] = True
    repo = RunRepository.from_config(config)
    seen = set()
    for ticker in ("UPRO", "AAPL"):
        for _ in range(2):
            result = run_clustering_workflow(ticker, config)
            artifacts = repo.get_run(result["run_id"])["artifacts"]
            assert len(artifacts) == 6
            paths = {Path(a["path"]) for a in artifacts}
            assert all(path.exists() and ticker.lower() in path.parts for path in paths)
            assert seen.isdisjoint(paths)
            seen.update(paths)
            stats = json.loads(Path(result["statistics_path"]).read_text())
            assert {"size", "percent", "overnight_delta_mean", "rsi_mean", "volume_ratio_mean"} <= stats[0].keys()
            assert sum(row["size"] for row in stats) == 160
