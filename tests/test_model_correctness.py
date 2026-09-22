import joblib
import numpy as np
import pandas as pd
import pytest

from qusa.model.backtest import ModelBacktester
from qusa.model.dataset import DatasetContractError, build_supervised_dataset
from qusa.model.evaluate import ModelEvaluator


class FixedProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, features):
        probabilities = self.probabilities[: len(features)]
        return np.column_stack((1 - probabilities, probabilities))

    def predict(self, features):
        return (self.probabilities[: len(features)] >= 0.5).astype(int)


def _write_backtest_inputs(tmp_path, closes, deltas, probabilities):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / "model.pkl"
    joblib.dump(
        {
            "model": FixedProbabilityModel(probabilities),
            "features": ["feature"],
            "threshold": 0.7,
        },
        model_path,
    )
    data_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(closes)),
            "close": closes,
            "overnight_delta": deltas,
            "feature": range(len(closes)),
        }
    ).to_csv(data_path, index=False)
    return model_path, data_path


def test_supervised_dataset_uses_next_session_and_rejects_ambiguous_dates():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4),
            "overnight_delta": [1.0, -2.0, 3.0, 4.0],
        }
    )

    dataset = build_supervised_dataset(data)

    assert dataset["target"].tolist() == [0, 1, 1]
    assert dataset["outcome_date"].tolist() == list(data["date"].iloc[1:])
    with pytest.raises(DatasetContractError, match="sorted"):
        build_supervised_dataset(data.iloc[[1, 0, 2, 3]])


def test_backtest_returns_are_price_normalized_and_scale_invariant(tmp_path):
    model_path, data_path = _write_backtest_inputs(
        tmp_path, [200.0, 204.0, 206.0], [0.0, 4.0, 2.0], [0.9, 0.1, 0.5]
    )
    backtester = ModelBacktester(model_path, data_path)
    results = backtester.run_backtest(1000, 1.0, 0.1)

    assert results["overnight_return"].tolist() == pytest.approx([0.02, 2 / 204])
    assert results["strategy_return"].tolist() == pytest.approx([0.018, -(2 / 204) - 0.002])

    scaled_model_path, scaled_data_path = _write_backtest_inputs(
        tmp_path / "scaled",
        [2000.0, 2040.0, 2060.0],
        [0.0, 40.0, 20.0],
        [0.9, 0.1, 0.5],
    )
    scaled = ModelBacktester(scaled_model_path, scaled_data_path).run_backtest(1000, 1.0, 0.1)
    assert scaled["strategy_return"].tolist() == pytest.approx(results["strategy_return"].tolist())


def test_backtest_invalid_outcomes_do_not_create_trades_and_first_loss_is_drawdown(tmp_path):
    model_path, data_path = _write_backtest_inputs(
        tmp_path, [100.0, 100.0, 100.0], [0.0, -1.0, 1.0], [0.9, 0.9, 0.5]
    )
    backtester = ModelBacktester(model_path, data_path)
    results = backtester.run_backtest(1000, 1.0, 0.0)
    assert results["strategy_return"].iloc[0] == pytest.approx(-0.01)
    assert backtester.calculate_metrics(1000)["max_draw_down"] == pytest.approx(-0.01)

    invalid_model_path, invalid_data_path = _write_backtest_inputs(
        tmp_path / "invalid", [0.0, 100.0, 100.0], [0.0, 1.0, 1.0], [0.9, 0.9, 0.5]
    )
    invalid = ModelBacktester(invalid_model_path, invalid_data_path).run_backtest(1000, 1.0, 0.0)
    assert invalid["invalid_outcome"].iloc[0]
    assert invalid["trade_count"].iloc[0] == 0


def test_evaluation_only_uses_outcomes_after_the_training_boundary(tmp_path):
    model_path = tmp_path / "model.pkl"
    joblib.dump(
        {
            "model": FixedProbabilityModel([0.9, 0.1]),
            "features": ["feature"],
            "threshold": 0.7,
            "target_version": "next_session_overnight_direction_v1",
            "model_id": "test-model",
            "dataset_metadata": {"training_outcome_end": "2025-01-03T00:00:00"},
        },
        model_path,
    )
    data_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "overnight_delta": [1.0, -1.0, 2.0, -2.0, 1.0],
            "feature": range(5),
        }
    ).to_csv(data_path, index=False)

    metrics = ModelEvaluator(model_path).evaluate(data_path)

    assert metrics["excluded_pre_training_outcomes"] == 2
    assert metrics["evaluation_outcome_start"] == "2025-01-04T00:00:00"
    assert metrics["model_id"] == "test-model"
