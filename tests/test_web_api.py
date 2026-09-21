from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from web_app import create_app
from web_app import api


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    for path in ("/health", "/api/health"):
        assert client.get(path).get_json() == {"status": "healthy"}


def test_ticker_discovery(client, monkeypatch, tmp_path):
    for name in ("UPRO_history.csv", "aapl_history.csv", "AAPL_history.csv", "AMZN_latest.csv"):
        (tmp_path / name).touch()
    (tmp_path / "FAKE_history.csv").mkdir()
    monkeypatch.setattr(api, "_load_app_config", lambda: {
        "data": {"paths": {"raw_data_dir": str(tmp_path)}}
    })
    assert client.get("/api/tickers").get_json() == {
        "success": True, "tickers": ["AAPL", "UPRO"]
    }


@pytest.mark.parametrize("endpoint", ["pipeline", "predictions"])
@pytest.mark.parametrize("body", [
    [], "UPRO", {}, {"ticker": None}, {"ticker": 123},
    {"ticker": "../UPRO"}, {"ticker": "UPRO", "fetch_latest": "false"},
])
def test_invalid_requests(client, endpoint, body):
    response = client.post(f"/api/{endpoint}/run", json=body)
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_malformed_json(client):
    response = client.post("/api/pipeline/run", data="{", content_type="application/json")
    assert response.status_code == 400


@pytest.mark.parametrize("volatility", [True, [], {}, "high", -1, float("inf"), float("nan")])
def test_invalid_volatility(client, volatility):
    response = client.post("/api/predictions/run", json={"ticker": "UPRO", "volatility": volatility})
    assert response.status_code == 400


def test_pipeline_delegation(client, monkeypatch, tmp_path):
    import qusa.services as services

    service = Mock(return_value={"success": True, "rows": np.int64(4),
                                "shape": (4, 2), "output_path": tmp_path / "output.csv"})
    monkeypatch.setattr(services, "run_feature_pipeline", service)
    response = client.post("/api/pipeline/run", json={"ticker": " upro "})
    assert response.status_code == 200
    assert response.get_json()["shape"] == [4, 2]
    service.assert_called_once_with(ticker="UPRO", fetch_latest=False, config_path=api.DEFAULT_CONFIG_PATH)


def test_prediction_serialization(client, monkeypatch):
    import qusa.services as services

    service = Mock(return_value={"success": True, "prediction": {
        "date": pd.Timestamp("2026-05-22"), "probability_up": np.float64(0.42),
        "atr_pct": float("nan"), "direction": "DOWN", "confidence": "LOW",
    }})
    monkeypatch.setattr(services, "make_latest_prediction", service)
    response = client.post("/api/predictions/run", json={"ticker": "upro", "volatility": 2.5, "fetch_latest": True})
    assert response.status_code == 200
    assert response.get_json()["prediction"]["date"] == "2026-05-22T00:00:00"
    assert response.get_json()["prediction"]["atr_pct"] is None
    service.assert_called_once_with(ticker="UPRO", fetch_latest=True, volatility_override=2.5, config_path=api.DEFAULT_CONFIG_PATH)


@pytest.mark.parametrize("error,status,code", [
    (ValueError("No data"), 500, "prediction_failed"),
    (FileNotFoundError("No model"), 404, "artifact_not_found"),
    (RuntimeError("Failed"), 500, "prediction_failed"),
])
def test_prediction_errors(client, monkeypatch, error, status, code):
    import qusa.services as services

    monkeypatch.setattr(services, "make_latest_prediction", Mock(side_effect=error))
    response = client.post("/api/predictions/run", json={"ticker": "UPRO"})
    assert response.status_code == status
    assert response.get_json()["success"] is False
    assert response.get_json()["code"] == code


@pytest.mark.parametrize("endpoint, service_name", [
    ("pipeline", "run_feature_pipeline"),
    ("predictions", "make_latest_prediction"),
])
def test_service_errors_do_not_expose_sensitive_values(
    client, monkeypatch, caplog, endpoint, service_name
):
    import qusa.services as services

    secret = "REVIEW_SENTINEL"
    error = RuntimeError(f"provider failed: https://example.test?apiKey={secret}")
    monkeypatch.setattr(services, service_name, Mock(side_effect=error))

    response = client.post(f"/api/{endpoint}/run", json={"ticker": "UPRO"})

    assert response.status_code == 500
    assert secret not in response.get_data(as_text=True)
    assert secret not in caplog.text


def test_history_filter_order_limit_and_nulls(client, monkeypatch, tmp_path):
    path = tmp_path / "predictions.csv"
    rows = [{"ticker": "UPRO", "timestamp": str(stamp), "probability_up": None}
            for stamp in pd.date_range("2026-01-01", periods=55)]
    rows.append({"ticker": "AAPL", "timestamp": "2027-01-01", "probability_up": 0.9})
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(api, "_prediction_log_path", lambda: path)
    response = client.get("/api/predictions/history?ticker=upro")
    assert response.status_code == 200
    history = response.get_json()["history"]
    assert len(history) == 50
    assert history[0]["timestamp"] == rows[54]["timestamp"]
    assert all(row["ticker"] == "UPRO" and row["probability_up"] is None for row in history)


def test_history_missing_and_empty(client, monkeypatch, tmp_path):
    path = tmp_path / "predictions.csv"
    monkeypatch.setattr(api, "_prediction_log_path", lambda: path)
    assert client.get("/api/predictions/history").get_json()["history"] == []
    path.touch()
    assert client.get("/api/predictions/history").get_json()["history"] == []


def test_history_does_not_read_text_log(client, monkeypatch, tmp_path):
    path = tmp_path / "prediction.log"
    path.write_text("This is a text log, not prediction data.")
    monkeypatch.setattr(api, "_load_app_config", lambda: {"prediction": {"log_file": str(path)}})
    assert client.get("/api/predictions/history").get_json()["history"] == []
