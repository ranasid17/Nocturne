"""Monte Carlo features based on daily log-return distributions."""

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class MonteCarloFeatures:
    """Generate deterministic, horizon-consistent Monte Carlo price features."""

    SEMANTICS_VERSION = "daily_log_return_v2"

    def __init__(self, config=None):
        self.config = config or {}
        self.window_size = int(self.config.get("window_size", 252))
        self.iterations = int(self.config.get("iterations", 1000))
        self.random_seed = int(self.config.get("random_seed", 42))
        self.min_data_threshold = int(self.config.get("min_data_threshold", 252))
        self.price_col = self.config.get("price_col", "close")
        self.horizons = self._validate_horizons(self.config.get("horizons", [1]))
        if self.window_size < 2 or self.iterations < 1 or self.min_data_threshold < 1:
            raise ValueError("Monte Carlo window, iterations, and threshold must be positive.")
        self.features = self.get_feature_names(horizons=self.horizons)

    @staticmethod
    def _validate_horizons(horizons):
        if not isinstance(horizons, (list, tuple)) or not horizons:
            raise ValueError("Monte Carlo horizons must be a non-empty list of positive integers.")
        if any(isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1 for horizon in horizons):
            raise ValueError("Monte Carlo horizons must be positive integers.")
        if len(set(horizons)) != len(horizons):
            raise ValueError("Monte Carlo horizons must not contain duplicates.")
        return list(horizons)

    def calculate_log_returns(self, prices):
        """Return daily log returns, ignoring non-positive prices."""

        safe_prices = pd.to_numeric(prices, errors="coerce").where(lambda values: values > 0)
        return np.log(safe_prices / safe_prices.shift(1))

    @staticmethod
    def _simulate_prices(current_price, daily_mean, daily_std, horizon, iterations, rng):
        """Simulate h-day terminal prices from a daily log-return distribution."""

        log_returns = rng.normal(
            loc=horizon * daily_mean,
            scale=np.sqrt(horizon) * daily_std,
            size=iterations,
        )
        return current_price * np.exp(log_returns)

    def _row_generator(self, row_position, horizon):
        # Row-specific streams preserve prior outputs when rows are appended.
        seed = np.random.SeedSequence([self.random_seed, int(row_position), horizon])
        return np.random.default_rng(seed)

    def add_all(self, df):
        """Fully and deterministically recompute all configured MC features."""

        if self.price_col not in df.columns:
            raise KeyError(f"Monte Carlo price column '{self.price_col}' is missing.")

        result = df.copy()
        for column in self.features:
            result[column] = np.nan
        if 1 in self.horizons:
            result["mc_1d_expected_value"] = np.nan

        if len(result) < self.min_data_threshold:
            return result

        log_returns = self.calculate_log_returns(result[self.price_col])
        rolling_mean = log_returns.rolling(window=self.window_size).mean()
        rolling_std = log_returns.rolling(window=self.window_size).std()
        prices = pd.to_numeric(result[self.price_col], errors="coerce")

        computed_rows = 0
        for position in range(self.min_data_threshold, len(result)):
            current_price = prices.iloc[position]
            daily_mean = rolling_mean.iloc[position]
            daily_std = rolling_std.iloc[position]
            if not (
                np.isfinite(current_price)
                and current_price > 0
                and np.isfinite(daily_mean)
                and np.isfinite(daily_std)
            ):
                continue

            row_index = result.index[position]
            for horizon in self.horizons:
                simulated_prices = self._simulate_prices(
                    current_price=current_price,
                    daily_mean=daily_mean,
                    daily_std=daily_std,
                    horizon=horizon,
                    iterations=self.iterations,
                    rng=self._row_generator(position, horizon),
                )
                quantiles = np.percentile(simulated_prices, [1, 5, 10, 50, 95])
                expected_value = float(np.mean(simulated_prices))
                result.loc[row_index, f"mc_{horizon}d_q1"] = quantiles[0]
                result.loc[row_index, f"mc_{horizon}d_q5"] = quantiles[1]
                result.loc[row_index, f"mc_{horizon}d_q10"] = quantiles[2]
                result.loc[row_index, f"mc_{horizon}d_q50"] = quantiles[3]
                result.loc[row_index, f"mc_{horizon}d_q95"] = quantiles[4]
                result.loc[row_index, f"mc_{horizon}d_return_pct"] = (
                    (expected_value - current_price) / current_price
                ) * 100
                result.loc[row_index, f"mc_{horizon}d_prob_breakeven"] = np.mean(
                    simulated_prices > current_price
                )
                if horizon == 1:
                    result.loc[row_index, "mc_1d_expected_value"] = expected_value
            computed_rows += 1

        logger.info("Computed Monte Carlo features for %s rows.", computed_rows)
        return result

    def add_mc_features(self, df, price_col=None):
        """Backward-compatible wrapper for older scripts."""

        original_price_col = self.price_col
        if price_col is not None:
            self.price_col = price_col
        try:
            return self.add_all(df)
        finally:
            self.price_col = original_price_col

    def validate_features(self, df):
        """Validate configured horizons, quantiles, probabilities, and finiteness."""

        report = {"total_rows": len(df), "valid_rows": 0, "nan_rows": 0, "errors": []}
        missing_columns = [column for column in self.features if column not in df.columns]
        if missing_columns:
            report["errors"].append("Missing Monte Carlo columns: " + ", ".join(missing_columns))
            report["nan_rows"] = len(df)
            return report

        valid_mask = df[self.features].notna().all(axis=1)
        report["valid_rows"] = int(valid_mask.sum())
        report["nan_rows"] = int(len(df) - report["valid_rows"])
        valid_data = df.loc[valid_mask]

        for horizon in self.horizons:
            prefix = f"mc_{horizon}d_"
            quantile_columns = [f"{prefix}{suffix}" for suffix in ("q1", "q5", "q10", "q50", "q95")]
            quantiles = valid_data[quantile_columns]
            ordering_violations = (
                (quantiles.iloc[:, 0] > quantiles.iloc[:, 1])
                | (quantiles.iloc[:, 1] > quantiles.iloc[:, 2])
                | (quantiles.iloc[:, 2] > quantiles.iloc[:, 3])
                | (quantiles.iloc[:, 3] > quantiles.iloc[:, 4])
            ).sum()
            if ordering_violations:
                report["errors"].append(
                    f"{horizon}d quantile ordering violations: {ordering_violations}"
                )

            probability = valid_data[f"{prefix}prob_breakeven"]
            probability_violations = ((probability < 0) | (probability > 1)).sum()
            if probability_violations:
                report["errors"].append(
                    f"{horizon}d probability out of bounds: {probability_violations}"
                )

        inf_count = np.isinf(df[self.features]).sum().sum()
        if inf_count:
            report["errors"].append(f"Infinite values detected: {inf_count}")
        return report

    @staticmethod
    def get_feature_names(horizons=None):
        if horizons is None:
            horizons = [1]
        suffixes = ("q1", "q5", "q10", "q50", "q95", "return_pct", "prob_breakeven")
        return [f"mc_{horizon}d_{suffix}" for horizon in horizons for suffix in suffixes]

    def get_feature_summary_string(self, df):
        summary = "MC Feature Statistics:\n" + "-" * 60 + "\n"
        for horizon in self.horizons:
            for column in (column for column in self.features if column.startswith(f"mc_{horizon}d_")):
                valid_data = df[column].dropna()
                if valid_data.empty:
                    summary += f"{column:30s}: No valid data\n"
                else:
                    summary += (
                        f"{column:30s}: [{valid_data.min():8.2f}, {valid_data.max():8.2f}], "
                        f"mean={valid_data.mean():8.2f}\n"
                    )
        return summary

    def print_feature_summary(self, df):
        print(self.get_feature_summary_string(df))
