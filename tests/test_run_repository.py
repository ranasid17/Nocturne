import csv
from contextlib import nullcontext

import pandas as pd
import pytest

from qusa.notifications.outbox import deliver_next_notification
from qusa.storage.database import schema_version
from qusa.storage.runs import DuplicateRunError, RunRepository, RunStateError


def _prediction(ticker="UPRO", timestamp="2026-01-02T00:00:00+00:00"):
    return {
        "ticker": ticker, "timestamp": timestamp, "date": "2026-01-01", "direction": "UP",
        "probability_up": 0.7, "confidence": "HIGH", "atr_pct": 1.2,
        "volatility_filter_triggered": False, "volatility_state": "clear",
        "volatility_threshold": 2.0, "readiness_status": "ready", "readiness": {"status": "ready"},
    }


def test_database_initialization_and_run_lifecycle(tmp_path):
    path = tmp_path / "history.sqlite3"
    repository = RunRepository(path)
    assert schema_version(path) == 1
    assert RunRepository(path).path == path
    run = repository.create_run("prediction", "UPRO", retry_key="request-1", owner_id="worker-a")
    with pytest.raises(DuplicateRunError):
        repository.create_run("prediction", "UPRO", retry_key="request-1")
    with pytest.raises(RunStateError):
        repository.transition_run(run["id"], "succeeded", owner_id="worker-a")
    repository.transition_run(run["id"], "running", owner_id="worker-a")
    repository.record_prediction(run["id"], _prediction())
    repository.transition_run(run["id"], "succeeded", owner_id="worker-a")
    assert repository.get_run(run["id"])["status"] == "succeeded"
    assert repository.list_predictions("UPRO")[0]["ticker"] == "UPRO"


def test_manual_runs_and_interrupted_recovery_are_distinct(tmp_path):
    repository = RunRepository(tmp_path / "history.sqlite3")
    first = repository.create_run("prediction", "UPRO")
    second = repository.create_run("prediction", "UPRO")
    repository.transition_run(second["id"], "running")
    assert first["id"] != second["id"]
    assert repository.recover_abandoned_runs() == 2
    assert repository.get_run(first["id"])["status"] == "interrupted"
    assert repository.get_run(second["id"])["status"] == "interrupted"


def test_legacy_import_is_dry_run_and_idempotent(tmp_path):
    source = tmp_path / "legacy.csv"
    source.write_text("ticker,timestamp,direction,probability_up\nUPRO,2026-01-01,UP,0.7\nUPRO,2026-01-01,DOWN,0.2\n")
    repository = RunRepository(tmp_path / "history.sqlite3")
    assert repository.import_legacy_csv(source, dry_run=True)["imported"] == 0
    assert repository.list_predictions() == []
    assert repository.import_legacy_csv(source)["imported"] == 2
    assert repository.import_legacy_csv(source)["imported"] == 0
    exported = tmp_path / "export.csv"
    assert repository.export_predictions_csv(exported) == 2
    with exported.open() as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_notification_worker_marks_delivery_and_never_retries_unknown(tmp_path, monkeypatch):
    repository = RunRepository(tmp_path / "history.sqlite3")
    run = repository.create_run("prediction", "UPRO")
    notification_id = repository.enqueue_notification(run["id"], ["desk@example.com"], {"ticker": "UPRO", "prediction": _prediction()})
    monkeypatch.setattr("qusa.notifications.outbox.send_prediction_email", lambda *args: {"sent": True, "recipients": ["desk@example.com"], "error": None})
    assert deliver_next_notification(repository, {})["id"] == notification_id
    assert repository.claim_notification() is None
    second = repository.enqueue_notification(run["id"], ["desk@example.com"], {"ticker": "UPRO"})
    assert repository.claim_notification()["id"] == second
    assert repository.recover_sending_notifications() == 1
    assert repository.claim_notification() is None


def test_prediction_service_persists_a_completed_run(tmp_path, monkeypatch):
    from qusa.services import prediction_service

    model_dir = tmp_path / "models"
    data_dir = tmp_path / "processed"
    model_dir.mkdir()
    data_dir.mkdir()
    (model_dir / "upro_model.pkl").touch()
    (data_dir / "UPRO_processed.csv").touch()
    repository = RunRepository(tmp_path / "history.sqlite3")
    config = {
        "data": {"paths": {"raw_data_dir": str(tmp_path), "processed_data_dir": str(data_dir)}},
        "model": {"output": {"model_output_path": str(model_dir)}},
        "backtest": {"volatility_filter": {"enabled": False}},
    }
    monkeypatch.setattr(prediction_service, "load_config", lambda path: config)
    monkeypatch.setattr(prediction_service, "ticker_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(prediction_service, "make_prediction", lambda *args, **kwargs: {
        "date": pd.Timestamp("2026-01-01"), "direction": "UP", "probability_up": 0.7,
        "confidence": "HIGH", "atr_pct": 1.1, "volatility_filter_triggered": False,
        "volatility_state": "clear", "volatility_threshold": 2.0, "model_id": "model-1",
    })
    monkeypatch.setattr(prediction_service.NyseSessionCalendar, "readiness_for_bar", lambda *args: {"status": "ready"})
    result = prediction_service.make_latest_prediction("UPRO", repository=repository)
    assert repository.get_run(result["run_id"])["status"] == "succeeded"
    assert repository.latest_prediction("UPRO")["feature_date"].startswith("2026-01-01")
