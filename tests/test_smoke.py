import importlib
from pathlib import Path

import pandas as pd

from qusa.features.monte_carlo import MonteCarloFeatures
from qusa.features.pipeline import FeaturePipeline
from qusa.model.train import get_safe_features, prepare_model_features
from qusa.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        "model": {"output": {"model_output_path": str(tmp_path / "models")}},
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
        ["run_model_pipeline.py", "TEST"],
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
