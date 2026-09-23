import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from qusa.features.monte_carlo import MonteCarloFeatures
from qusa.features.pipeline import FeaturePipeline
from qusa.model.predict import LivePredictor
from qusa.model.train import get_safe_features
from qusa.services.pipeline_service import _build_feature_pipeline
from qusa.services.research_service import _model_config


def test_monte_carlo_daily_log_return_distribution_and_zero_volatility():
    mean = 0.001
    std = 0.02
    horizon = 7
    prices = MonteCarloFeatures._simulate_prices(
        100.0, mean, std, horizon, 100_000, np.random.default_rng(4)
    )
    simulated_log_returns = np.log(prices / 100.0)

    assert simulated_log_returns.mean() == pytest.approx(horizon * mean, abs=0.0003)
    assert simulated_log_returns.var() == pytest.approx(horizon * std**2, rel=0.03)

    deterministic = MonteCarloFeatures._simulate_prices(
        100.0, mean, 0.0, horizon, 50, np.random.default_rng(8)
    )
    assert np.unique(deterministic) == pytest.approx([100.0 * np.exp(horizon * mean)])


def test_monte_carlo_recomputes_deterministically_without_global_rng_side_effects():
    config = {
        "window_size": 5,
        "min_data_threshold": 5,
        "iterations": 100,
        "random_seed": 7,
        "horizons": [3, 7],
    }
    input_data = pd.DataFrame({"close": np.linspace(100.0, 125.0, 18)})
    calculator = MonteCarloFeatures(config)
    first = calculator.add_all(input_data.iloc[:14])
    second = calculator.add_all(input_data)

    pd.testing.assert_frame_equal(
        first[calculator.features], second.iloc[:14][calculator.features]
    )
    assert calculator.validate_features(second)["errors"] == []
    assert not any(column.startswith("mc_1d_") for column in calculator.features)

    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)
    calculator.add_all(input_data)
    assert np.random.random(4) == pytest.approx(expected)


def test_active_manifest_matches_disabled_and_alternate_horizon_features():
    disabled = get_safe_features(include_monte_carlo=False)
    alternate = get_safe_features(include_monte_carlo=True, mc_horizons=[3, 7])

    assert not any(column.startswith("mc_") for column in disabled)
    assert "mc_3d_q50" in alternate
    assert "mc_7d_q50" in alternate
    assert "mc_1d_q50" not in alternate
    assert len(alternate) == len(set(alternate))


def test_feature_service_forwards_all_custom_feature_windows():
    pipeline = _build_feature_pipeline(
        {
            "analysis": {"abnormal_threshold": 2.0},
            "features": {
                "rsi_window": 9,
                "vwap_window": 11,
                "vol_regime_short_window": 4,
                "vol_regime_long_window": 15,
                "advanced_vol_window": 12,
            },
        }
    )

    assert pipeline.technical_indicators.period_rsi == 9
    assert pipeline.volatility_calculator.vwap_window == 11
    assert pipeline.volatility_calculator.vol_regime_short == 4
    assert pipeline.volatility_calculator.vol_regime_long == 15
    assert pipeline.volatility_calculator.advanced_vol_window == 12


def test_training_config_forwards_tuning_and_rejects_malformed_grids():
    config = {"model": {"parameters": {"tuning": {"enabled": True, "param_grid": {"max_depth": [3, 5]}}}}}
    assert _model_config(config)["tuning"]["param_grid"] == {"max_depth": [3, 5]}

    malformed = {"model": {"parameters": {"tuning": {"enabled": True, "param_grid": {"bad": [1]}}}}}
    with pytest.raises(ValueError, match="Unsupported"):
        _model_config(malformed)


def _write_predictor_bundle(tmp_path):
    features = ["feature", "atr_pct"]
    training = pd.DataFrame({"feature": [0.0, 1.0], "atr_pct": [1.0, 1.0]})
    model = DecisionTreeClassifier(random_state=1).fit(training, [0, 1])
    path = tmp_path / "model.pkl"
    joblib.dump({"model": model, "features": features, "threshold": 0.7}, path)
    return path


def test_predictor_enforces_features_and_reports_unavailable_or_blocked_volatility(tmp_path):
    predictor = LivePredictor(_write_predictor_bundle(tmp_path))

    unavailable = predictor.predict(
        pd.DataFrame({"date": ["2025-01-01"], "feature": [np.inf], "atr_pct": [np.nan]}),
        volatility_filter={"enabled": True, "max_atr_pct": 2.0},
    )
    assert unavailable["volatility_state"] == "unavailable"
    assert unavailable["volatility_filter_triggered"] is False
    assert unavailable["atr_pct"] is None

    blocked = predictor.predict(
        pd.DataFrame({"date": ["2025-01-01"], "feature": [1.0], "atr_pct": [2.5]}),
        volatility_filter={"enabled": True, "max_atr_pct": 2.0},
    )
    assert blocked["volatility_state"] == "blocked"
    assert blocked["volatility_filter_triggered"] is True

    with pytest.raises(ValueError, match="missing model features"):
        predictor.predict(pd.DataFrame({"feature": [1.0]}))


def test_predictor_rejects_a_bundle_with_an_incompatible_feature_schema(tmp_path):
    path = _write_predictor_bundle(tmp_path)
    bundle = joblib.load(path)
    bundle["features"] = ["feature"]
    joblib.dump(bundle, path)

    with pytest.raises(ValueError, match="feature schema"):
        LivePredictor(path)
