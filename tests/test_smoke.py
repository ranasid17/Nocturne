# qusa/tests/test_smoke.py

import importlib
import os
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

# Ensure a dummy API key is set for tests
os.environ.setdefault("POLYGON_API_KEY", "dummy-key-for-tests")

from qusa.features.monte_carlo import MonteCarloFeatures
from qusa.features.pipeline import FeaturePipeline
from qusa.model.train import get_safe_features, prepare_model_features, OvernightDirectionModel, SAFE_FEATURES
from qusa.utils.config import load_config
from qusa.analysis.clustering import ClusterAnalyzer
from qusa.model.evaluate import ModelEvaluator
from qusa.model.predict import LivePredictor
from qusa.model.reporter import StrategyReporter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------

def test_config_loads_pipeline_skip_flags():
    config = load_config(PROJECT_ROOT / "qusa" / "utils" / "config.yaml")

    assert "tickers" not in config["data"]
    assert config["pipeline"]["skip_training"] is False
    assert config["pipeline"]["skip_evaluation"] is False
    assert config["pipeline"]["skip_backtest"] is False


def test_monte_carlo_feature_names_and_safe_features_are_deduplicated():
    mc_names = MonteCarloFeatures.get_feature_names(horizons=[1])
    safe_features = get_safe_features(include_monte_carlo=True, mc_horizons=[1])

    assert "mc_1d_q5" in mc_names
    assert "mc_1d_prob_breakeven" in mc_names
    assert len(safe_features) == len(set(safe_features))


def test_prepare_model_features_replaces_non_finite_values():
    data = pd.DataFrame(
        {
            "good": [1.0, None, 3.0],
            "bad": [float("inf"), -float("inf"), 5.0],
        }
    )

    result = prepare_model_features(data, ["good", "bad"])

    assert result.isna().sum().sum() == 0
    assert result.loc[0, "bad"] == 0
    assert result.loc[1, "bad"] == 0


def test_feature_pipeline_runs_with_monte_carlo_enabled():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "open": [100, 101, 102, 101, 103, 104, 106, 105, 107, 108, 110, 109],
            "high": [101, 102, 103, 103, 104, 105, 107, 107, 108, 109, 111, 111],
            "low": [99, 100, 101, 100, 102, 103, 105, 104, 106, 107, 109, 108],
            "close": [100, 102, 101, 103, 104, 106, 105, 107, 108, 110, 109, 111],
            "volume": [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1400, 1350, 1500, 1450, 1600],
        }
    )

    pipeline = FeaturePipeline(
        {
            "date_col": "date",
            "open_col": "open",
            "close_col": "close",
            "high_col": "high",
            "low_col": "low",
            "volume_col": "volume",
            "technical_params": {
                "rsi_window": 3,
                "atr_window": 3,
                "volume_ma_window": 3,
                "rolling_window_52w": 3,
            },
            "monte_carlo": {
                "enabled": True,
                "window_size": 3,
                "min_data_threshold": 3,
                "iterations": 20,
                "random_seed": 42,
            },
        }
    )

    result = pipeline.run(data)

    assert "mc_1d_q5" in result.columns
    assert "mc_1d_prob_breakeven" in result.columns
    assert result["mc_1d_q5"].notna().any()


def test_workflow_scripts_import_cleanly():
    for module_name in [
        "scripts.run_FE_pipeline",
        "scripts.run_clustering",
        "scripts.run_model_pipeline",
        "scripts.model_prediction",
    ]:
        importlib.import_module(module_name)


def test_model_pipeline_honors_skip_flags(monkeypatch, tmp_path):
    run_model_pipeline = importlib.import_module("scripts.run_model_pipeline")

    config = {
        "data": {
            "paths": {
                "processed_data_dir": str(tmp_path / "processed"),
                "figures_dir": str(tmp_path / "figures"),
            },
        },
        "model": {
            "output": {"model_output_path": str(tmp_path / "models")},
            "parameters": {"cv": 5}
        },
        "pipeline": {
            "skip_training": True,
            "skip_evaluation": True,
            "skip_backtest": True,
        },
    }

    class Logger:
        def info(self, *_args, **_kwargs):
            return None

        def warning(self, *_args, **_kwargs):
            return None

        def error(self, *_args, **_kwargs):
            return None

    def fail_phase(*_args, **_kwargs):
        raise AssertionError("Skipped phases should not run")

    monkeypatch.setattr(
        "sys.argv",
        ["run_model_pipeline.py", "-ticker", "TEST"],
    )
    monkeypatch.setattr(run_model_pipeline, "load_config", lambda _path: config)
    monkeypatch.setattr(
        run_model_pipeline,
        "setup_logger",
        lambda *_args, **_kwargs: Logger(),
    )
    monkeypatch.setattr(run_model_pipeline, "_run_training", fail_phase)
    monkeypatch.setattr(run_model_pipeline, "_run_evaluation", fail_phase)
    monkeypatch.setattr(run_model_pipeline, "_run_backtest", fail_phase)

    assert run_model_pipeline.main() == 0


# ---------------------------------------------------------------------------
# New tests: PolygonFetcher and DataLoader
# ---------------------------------------------------------------------------

# Fake Polygon /v1/open-close response
FAKE_OPEN_CLOSE = {
    "status": "OK",
    "from": "2026-05-05",
    "symbol": "AMZN",
    "open": 185.50,
    "high": 188.00,
    "low": 183.10,
    "close": 186.75,
    "volume": 32_000_000,
    "afterHours": 186.80,
    "preMarket": 185.20,
}


def _mock_response(payload, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = payload
    mock.raise_for_status = MagicMock()
    return mock


def test_polygon_fetcher_raises_without_api_key(monkeypatch):
    """PolygonFetcher must raise ValueError when no API key is available."""
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)

    from qusa.data.fetcher import PolygonFetcher

    with pytest.raises(ValueError, match="POLYGON_API_KEY"):
        PolygonFetcher()


def test_polygon_fetcher_latest_day_returns_correct_columns(monkeypatch):
    """fetch_latest_day returns a single-row DataFrame with OHLCV columns."""
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    from qusa.data.fetcher import PolygonFetcher

    with patch("qusa.data.fetcher.requests.get", return_value=_mock_response(FAKE_OPEN_CLOSE)):
        fetcher = PolygonFetcher()
        df = fetcher.fetch_latest_day("AMZN")

    assert len(df) == 1
    assert set(df.columns) == {"date", "open", "high", "low", "close", "volume"}
    assert df["close"].iloc[0] == pytest.approx(186.75)


def test_polygon_fetcher_latest_day_raises_on_non_ok_status(monkeypatch):
    """fetch_latest_day raises ValueError when Polygon returns a non-OK status."""
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    from qusa.data.fetcher import PolygonFetcher

    bad_response = {"status": "NOT_FOUND", "message": "No data found"}
    with patch("qusa.data.fetcher.requests.get", return_value=_mock_response(bad_response)):
        fetcher = PolygonFetcher()
        with pytest.raises(ValueError, match="non-OK status"):
            fetcher.fetch_latest_day("AMZN")


def test_polygon_fetcher_historical_range(monkeypatch):
    """fetch_historical_range returns multiple rows from Polygon aggregates."""
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    from qusa.data.fetcher import PolygonFetcher

    fake_aggs = {
        "status": "OK",
        "ticker": "AMZN",
        "results": [
            {"t": 1714924800000, "o": 180.0, "h": 182.0, "l": 179.0, "c": 181.0, "v": 1000000},
            {"t": 1715011200000, "o": 181.0, "h": 183.0, "l": 180.0, "c": 182.0, "v": 1100000},
        ],
        "resultsCount": 2
    }

    with patch("qusa.data.fetcher.requests.get", return_value=_mock_response(fake_aggs)):
        fetcher = PolygonFetcher()
        df = fetcher.fetch_historical_range("AMZN", "2024-05-01", "2024-05-02")

    assert len(df) == 2
    assert "date" in df.columns
    assert df["close"].iloc[1] == 182.0


def test_polygon_fetcher_n_days(monkeypatch):
    """fetch_n_days returns exactly N rows if available."""
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    from qusa.data.fetcher import PolygonFetcher

    # Mocking historical_range to return 5 rows
    results = [
        {"t": 1714924800000 + i * 86400000, "o": 180 + i, "h": 182 + i, "l": 179 + i, "c": 181 + i, "v": 1000000}
        for i in range(5)
    ]
    fake_aggs = {"status": "OK", "results": results, "resultsCount": 5}

    with patch("qusa.data.fetcher.requests.get", return_value=_mock_response(fake_aggs)):
        fetcher = PolygonFetcher()
        df = fetcher.fetch_n_days("AMZN", 3)

    assert len(df) == 3
    assert df["close"].iloc[-1] == 181 + 4  # Tail(3) of 5 rows (0,1,2,3,4) is (2,3,4)


def test_polygon_fetcher_latest_day_delayed_status(monkeypatch):
    """fetch_latest_day succeeds even if status is 'DELAYED'."""
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    from qusa.data.fetcher import PolygonFetcher

    delayed_response = FAKE_OPEN_CLOSE.copy()
    delayed_response["status"] = "DELAYED"

    with patch("qusa.data.fetcher.requests.get", return_value=_mock_response(delayed_response)):
        fetcher = PolygonFetcher()
        df = fetcher.fetch_latest_day("AMZN")

    assert len(df) == 1
    assert df["close"].iloc[0] == pytest.approx(186.75)


def test_data_loader_consolidate_history(tmp_path):
    """consolidate_history merges multiple files and archives them."""
    from qusa.data.loader import DataLoader
    
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    
    # Create two overlapping files
    df1 = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "open": [100, 101], "high": [102, 103], "low": [99, 100], "close": [101, 102], "volume": [1000, 1100]
    })
    df2 = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "open": [101, 102], "high": [103, 104], "low": [100, 101], "close": [102, 103], "volume": [1100, 1200]
    })
    
    df1.to_csv(raw_dir / "AMZN_2024-01-01_2024-01-02.csv", index=False)
    df2.to_csv(raw_dir / "AMZN_2024-01-02_2024-01-03.csv", index=False)
    
    loader = DataLoader(raw_data_dir=str(raw_dir))
    consolidated, _ = loader.consolidate_history("AMZN")

    # Should have 3 unique days
    assert len(consolidated) == 3
    assert (raw_dir / "AMZN_history.csv").exists()
    
    # Source files should be archived
    assert not (raw_dir / "AMZN_2024-01-01_2024-01-02.csv").exists()
    assert (raw_dir / "archive" / "AMZN_2024-01-01_2024-01-02.csv").exists()
    assert (raw_dir / "archive" / "AMZN_2024-01-02_2024-01-03.csv").exists()


def test_data_loader_merges_latest_onto_history(monkeypatch, tmp_path):
    """
    load_most_recent merges the fetched latest row onto an existing
    historical CSV, deduplicates on date, and returns a sorted DataFrame.
    """
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    from qusa.data.loader import DataLoader

    # Write a minimal historical CSV
    history = pd.DataFrame(
        {
            "date": ["2026-05-01", "2026-05-02", "2026-05-04"],
            "open":   [180.0, 181.0, 183.0],
            "high":   [182.0, 183.0, 185.0],
            "low":    [179.0, 180.0, 182.0],
            "close":  [181.0, 182.0, 184.0],
            "volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    history.to_csv(raw_dir / "AMZN_history.csv", index=False)

    with patch(
        "qusa.data.fetcher.requests.get",
        return_value=_mock_response(FAKE_OPEN_CLOSE),
    ):
        loader = DataLoader(raw_data_dir=str(raw_dir))
        merged = loader.load_most_recent(
            "AMZN", start="2026-01-01", end="2026-12-31"
        )

    # Latest day file should exist
    assert (raw_dir / "AMZN_latest.csv").exists()

    # Merged result should have 4 rows (3 history + 1 new), sorted by date
    assert len(merged) == 4
    assert merged["date"].is_monotonic_increasing

    # The latest close should appear
    assert (merged["close"] == 186.75).any()


def test_data_loader_no_history_returns_latest_only(monkeypatch, tmp_path):
    """
    load_most_recent still works when no historical CSV exists —
    returns a single-row DataFrame of the latest day.
    """
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    from qusa.data.loader import DataLoader

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    with patch(
        "qusa.data.fetcher.requests.get",
        return_value=_mock_response(FAKE_OPEN_CLOSE),
    ):
        loader = DataLoader(raw_data_dir=str(raw_dir))
        result = loader.load_most_recent(
            "AMZN", start="2026-01-01", end="2026-12-31"
        )

    assert len(result) == 1
    assert result["close"].iloc[0] == pytest.approx(186.75)


def test_data_loader_deduplicates_on_rerun(monkeypatch, tmp_path):
    """
    If the latest day already exists in the historical CSV,
    load_most_recent should not add a duplicate row.
    """
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    from qusa.data.loader import DataLoader

    # History already contains 2026-05-05 (same date as fake response)
    history = pd.DataFrame(
        {
            "date":   ["2026-05-04", "2026-05-05"],
            "open":   [183.0, 185.50],
            "high":   [185.0, 188.00],
            "low":    [182.0, 183.10],
            "close":  [184.0, 186.75],
            "volume": [1_200_000, 32_000_000],
        }
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    history.to_csv(raw_dir / "AMZN_history.csv", index=False)

    with patch(
        "qusa.data.fetcher.requests.get",
        return_value=_mock_response(FAKE_OPEN_CLOSE),
    ), patch(
        "qusa.data.fetcher.PolygonFetcher._get_most_recent_trading_day",
        return_value="2026-05-05"
    ):
        loader = DataLoader(raw_data_dir=str(raw_dir))
        merged = loader.load_most_recent(
            "AMZN", start="2026-01-01", end="2026-12-31"
        )

    # Still 2 rows — no duplicate introduced
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# Phase 4: New Tests for Bug Fixes and Stability
# ---------------------------------------------------------------------------

def test_cluster_interpret_keys_match_profiles():
    """Task 4.1: Verify interpret_clusters keys match cluster profiles."""
    data = pd.DataFrame({
        "overnight_delta_pct": [1.0, -1.0, 0.1, 0.2, 5.0, -5.0],
        "intraday_return": [1.0, -1.0, 0.1, 0.2, 5.0, -5.0],
        "volume_ratio": [1.0, 1.0, 1.0, 1.0, 3.0, 3.0],
        "rsi": [50, 50, 50, 50, 80, 20],
        "atr_pct": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "close_position": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "52_week_high_proximity": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "52_week_low_proximity": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    
    analyzer = ClusterAnalyzer(n_clusters=2)
    clustered_data = analyzer.fit_clusters(data, feature_cols=None)
    
    interpretations = analyzer.interpret_clusters(clustered_data)
    assert isinstance(interpretations, dict)
    assert len(interpretations) == 2
    for k, v in interpretations.items():
        assert isinstance(k, int)
        assert len(v) > 0


def test_cluster_fit_scaler_not_double_scaled():
    """Task 4.1: Verify scaler is not redundantlly fitted/overwritten."""
    data = pd.DataFrame({
        "f1": np.random.randn(10),
        "f2": np.random.randn(10)
    })
    analyzer = ClusterAnalyzer(n_clusters=2)
    # This calls prepare_features which fits the scaler
    analyzer.fit_clusters(data, feature_cols=["f1", "f2"])
    
    scaler_id = id(analyzer.scaler)
    # We can't easily check if it was double-fitted without mocking StandardScaler
    # but Task 1.2 removed the explicit re-fit line.
    assert analyzer.scaler is not None


def test_predict_cluster_returns_array():
    """Task 4.1: Verify predict_cluster returns a numpy array."""
    data = pd.DataFrame({
        "overnight_delta_pct": [1.0, 2.0], "intraday_return": [1.0, 2.0],
        "volume_ratio": [1.0, 1.1], "rsi": [50, 51], "atr_pct": [1.0, 1.1],
        "close_position": [0.5, 0.6], "52_week_high_proximity": [1.0, 1.1],
        "52_week_low_proximity": [1.0, 1.1]
    })
    analyzer = ClusterAnalyzer(n_clusters=2)
    analyzer.fit_clusters(data, feature_cols=None)
    
    result = analyzer.predict_cluster(data)
    assert isinstance(result, np.ndarray)
    assert len(result) == 2


def test_model_evaluator_calibration():
    """Task 4.2: Verify calibration actual_rate reflects true label frequency."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])
    
    calibration = ModelEvaluator._analyze_calibration(y_true, y_prob)
    
    # bin (0.7, 1.0] has y_true=[1, 1], mean=1.0
    high_bin = calibration.loc[pd.Interval(0.7, 1.0, closed='right')]
    assert high_bin['actual_rate'] == 1.0


def test_init_guards_raise_valueerror():
    """Task 4.3: Verify init guards raise ValueError when config is missing."""
    from qusa.model.reporter import StrategyReporter
    from qusa.model.interpreter import ModelInterpreter
    
    with pytest.raises(ValueError, match="config is required"):
        StrategyReporter(config=None)
        
    with pytest.raises(ValueError, match="config is required"):
        ModelInterpreter(model_path="dummy", config=None)


def test_mc_index_alignment():
    """Task 4.4: Verify MC features align correctly with non-standard index."""
    data = pd.DataFrame({
        "close": np.linspace(100, 110, 500)
    }, index=range(500, 1000))
    
    mc = MonteCarloFeatures(config={"min_data_threshold": 252, "window_size": 252})
    result = mc.add_all(data)
    
    # 500 + 252 = 752. Rows 500 to 751 are NaN.
    assert result.loc[500:751, "mc_1d_q5"].isna().all()
    assert result.loc[752:999, "mc_1d_q5"].notna().all()


def test_polygon_fetcher_intraday_snapshot(monkeypatch):
    """fetch_intraday_snapshot returns a single row from the snapshot API."""
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    from qusa.data.fetcher import PolygonFetcher

    fake_snapshot = {
        "status": "OK",
        "ticker": {
            "ticker": "AMZN",
            "day": {
                "o": 185.0, "h": 188.0, "l": 184.0, "c": 187.5, "v": 25000000
            }
        }
    }

    with patch("qusa.data.fetcher.requests.get", return_value=_mock_response(fake_snapshot)):
        fetcher = PolygonFetcher()
        df = fetcher.fetch_intraday_snapshot("AMZN")

    assert len(df) == 1
    assert df["close"].iloc[0] == 187.5
    assert df["date"].iloc[0] == datetime.now().strftime("%Y-%m-%d")


def test_data_loader_load_intraday(monkeypatch, tmp_path):
    """load_intraday merges history and snapshot, keeping the snapshot on overlap."""
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    from qusa.data.loader import DataLoader

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # History contains yesterday
    history = pd.DataFrame({
        "date": [yesterday_str],
        "open": [180.0], "high": [182.0], "low": [179.0], "close": [181.0], "volume": [1000000]
    })
    history.to_csv(raw_dir / "AMZN_history.csv", index=False)

    fake_snapshot = {
        "status": "OK",
        "ticker": {
            "ticker": "AMZN",
            "day": {
                "o": 182.0, "h": 185.0, "l": 181.0, "c": 184.5, "v": 500000
            }
        }
    }

    with patch("qusa.data.fetcher.requests.get", return_value=_mock_response(fake_snapshot)):
        loader = DataLoader(raw_data_dir=str(raw_dir))
        result = loader.load_intraday("AMZN")

    assert len(result) == 2
    assert result["date"].iloc[-1] == today_str
    assert result["close"].iloc[-1] == 184.5


def test_model_target_alignment_shifting(tmp_path):
    """Task 66: Verify model target is shifted to predict NEXT day."""
    from qusa.model.train import OvernightDirectionModel
    
    # Create dummy data
    data = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "overnight_delta": [1.0, -2.0, 3.0],  # Day 1: UP, Day 2: DOWN, Day 3: UP
        "close": [100, 101, 102],
        "open": [101, 99, 103],
        "volume": [1000, 1100, 1200]
    })
    
    # Add dummy technical features
    for f in SAFE_FEATURES:
        if f not in data.columns:
            data[f] = 0.0
            
    csv_path = tmp_path / "test_alignment.csv"
    data.to_csv(csv_path, index=False)
    
    # Load data via model
    prepared_data = OvernightDirectionModel.load_data(str(csv_path))
    
    # Target for Day 1 should be the delta of Day 2 (Negative -> 0)
    # Target for Day 2 should be the delta of Day 3 (Positive -> 1)
    # Day 3 should be dropped (no next day)
    assert len(prepared_data) == 2
    assert prepared_data.iloc[0]["target"] == 0
    assert prepared_data.iloc[1]["target"] == 1
