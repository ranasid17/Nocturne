"""Public wrapper and probability contracts kept while removing legacy helpers."""

import csv
import importlib
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from qusa.model.probability import probability_of_up
from qusa.services.research_service import run_model_workflow
from qusa.storage.runs import RunRepository


@pytest.mark.parametrize(
    "classes, rows, expected",
    [
        ([0, 1], [[0.8, 0.2], [0.3, 0.7]], [0.2, 0.7]),
        ([1, 0], [[0.8, 0.2], [0.3, 0.7]], [0.8, 0.3]),
        ([0], [[1.0], [1.0]], [0.0, 0.0]),
        ([1], [[1.0], [1.0]], [1.0, 1.0]),
    ],
)
def test_probability_of_up_respects_class_order_and_one_class_models(classes, rows, expected):
    model = Mock(classes_=np.array(classes))
    model.predict_proba.return_value = np.array(rows)
    assert probability_of_up(model, np.zeros((2, 1))) == pytest.approx(expected)


def _minimal_config(tmp_path, skip=False):
    return {
        "data": {"paths": {
            "processed_data_dir": str(tmp_path / "processed"),
            "figures_dir": str(tmp_path / "figures"),
        }},
        "model": {"output": {"model_output_path": str(tmp_path / "models")}, "parameters": {}},
        "backtest": {"volatility_filter": {"enabled": False}},
        "pipeline": {
            "skip_training": skip, "skip_evaluation": skip, "skip_backtest": skip,
        },
    }


@pytest.mark.parametrize("skip, expected_exit", [(False, 1), (True, 0)])
def test_model_cli_and_service_agree_on_missing_data_and_skip(tmp_path, monkeypatch, skip, expected_exit):
    script = importlib.import_module("scripts.run_model_pipeline")
    config = _minimal_config(tmp_path, skip)
    direct = run_model_workflow("UPRO", config)
    logger = Mock()
    monkeypatch.setattr(script, "load_config", lambda: config)
    monkeypatch.setattr(script, "setup_logger", lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(sys, "argv", ["run_model_pipeline.py", "-ticker", "UPRO"])
    assert script.main() == expected_exit
    assert direct["success"] is (expected_exit == 0)
    assert {phase["status"] for phase in direct["phases"].values()} == (
        {"skipped"} if skip else {"failed"}
    )
    assert logger.info.call_count >= 4


def test_model_cli_propagates_phase_exception_as_failure(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.run_model_pipeline")
    config = _minimal_config(tmp_path)
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "UPRO_processed.csv").write_text("date\n2026-01-01\n")
    monkeypatch.setattr("qusa.services.research_service.train_model", Mock(side_effect=ValueError("invalid fixture")))
    with pytest.raises(ValueError):
        run_model_workflow("UPRO", config)
    monkeypatch.setattr(script, "load_config", lambda: config)
    monkeypatch.setattr(script, "setup_logger", lambda *_args, **_kwargs: Mock())
    monkeypatch.setattr(sys, "argv", ["run_model_pipeline.py", "-ticker", "UPRO"])
    assert script.main() == 1


@pytest.mark.parametrize("success, expected_exit", [(True, 0), (False, 1)])
def test_clustering_cli_keeps_ticker_flag_and_exit_status(monkeypatch, success, expected_exit):
    script = importlib.import_module("scripts.run_clustering")
    result = {"success": success, "output_path": "clustered.csv", "statistics_path": "stats.json"}
    workflow = Mock(return_value=result)
    monkeypatch.setattr(script, "load_config", lambda: {})
    monkeypatch.setattr(script, "setup_logger", lambda *_args, **_kwargs: Mock())
    monkeypatch.setattr(script, "run_clustering_workflow", workflow)
    monkeypatch.setattr(sys, "argv", ["run_clustering.py", "-ticker", "upro"])
    assert script.main() == expected_exit
    workflow.assert_called_once_with("UPRO", {})


def test_legacy_import_export_keeps_row_order_and_null_values(tmp_path):
    source = tmp_path / "legacy.csv"
    source.write_text(
        "ticker,timestamp,date,direction,volatility_filter_triggered\n"
        "UPRO,2026-01-01T00:00:00,2025-12-31,UP,\n"
        "UPRO,2026-01-02T00:00:00,2026-01-01,DOWN,true\n"
    )
    repository = RunRepository(tmp_path / "runs.sqlite3")
    assert repository.import_legacy_csv(source)["imported"] == 2
    assert repository.import_legacy_csv(source)["imported"] == 0
    rows = repository.list_predictions("UPRO")
    assert [row["date"] for row in rows] == ["2026-01-01", "2025-12-31"]
    assert rows[1]["volatility_filter_triggered"] is None
    destination = tmp_path / "export.csv"
    assert repository.export_predictions_csv(destination) == 2
    with destination.open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert [row["date"] for row in exported] == ["2026-01-01", "2025-12-31"]
