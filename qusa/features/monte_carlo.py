# qusa/features/monte_carlo.py

"""
Monte Carlo feature generation for stock price forecasting.
Uses geometric Brownian motion to simulate future price paths.
"""

import numpy as np
import pandas as pd
from datetime import datetime


class MonteCarloFeatures:
    """
    Calculate Monte Carlo simulation features for financial time series data.
    """

    def __init__(self, config=None):
        """
        Class constructor.

        Parameters:
            config (dict): Configuration dictionary with MC settings
        """
        self.config = config or {}

        # Extract parameters from config with defaults
        self.window_size = self.config.get("window_size", 252)
        self.iterations = self.config.get("iterations", 1000)
        self.random_seed = self.config.get("random_seed", 42)
        self.min_data_threshold = self.config.get("min_data_threshold", 252)
        self.price_col = self.config.get("price_col", "close")
        self.features = self.config.get(
            "features",
            [
                "mc_1d_q1",
                "mc_1d_q5",
                "mc_1d_q10",
                "mc_1d_q50",
                "mc_1d_q95",
                "mc_1d_return_pct",
                "mc_1d_prob_breakeven",
            ],
        )

    def calculate_log_returns(self, prices):
        """
        Calculate log returns from price series.

        Parameters:
            prices (pd.Series): Price series

        Returns:
            pd.Series: Log returns
        """
        return np.log(prices / prices.shift(1))

    def calculate_drift(self, log_returns):
        """
        Calculate drift (expected return) from log returns.

        Parameters:
            log_returns (pd.Series): Log returns series

        Returns:
            float: Drift value
        """
        mean_return = log_returns.mean()
        variance = log_returns.var()
        drift = mean_return - (0.5 * variance)

        return drift

    def simulate_price_paths(self, current_price, drift, volatility, days=1):
        """
        Simulate future price paths using geometric Brownian motion.

        Parameters:
            current_price (float): Starting price
            drift (float): Expected return (drift)
            volatility (float): Historical volatility (std dev of log returns)
            days (int): Number of days to simulate

        Returns:
            np.ndarray: Array of simulated end prices (shape: iterations,)
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Daily timestep
        dt = 1 / 252

        # Generate random shocks
        shocks = np.random.normal(size=(days, self.iterations))

        # Calculate daily returns
        daily_returns = np.exp(drift * dt + volatility * np.sqrt(dt) * shocks)

        # Calculate end prices (product of all daily returns)
        cumulative_returns = np.prod(daily_returns, axis=0)
        end_prices = current_price * cumulative_returns

        return end_prices

    def calculate_mc_features_for_window(self, price_window):
        """
        Calculate MC features for a single rolling window.

        Parameters:
            price_window (pd.Series): Rolling window of historical prices

        Returns:
            dict: Dictionary of MC feature values
        """
        try:
            # Validate window size
            if len(price_window) < self.window_size:
                return None

            # Get current price (last price in window)
            current_price = price_window.iloc[-1]

            # Calculate log returns
            log_returns = self.calculate_log_returns(price_window).dropna()

            # Check for sufficient data
            if len(log_returns) < 2:
                return None

            # Calculate drift and volatility
            drift = self.calculate_drift(log_returns)
            volatility = log_returns.std()

            # Handle zero volatility edge case
            if volatility == 0 or np.isnan(volatility):
                return None

            # Simulate price paths (1 day ahead)
            simulated_prices = self.simulate_price_paths(
                current_price=current_price, drift=drift, volatility=volatility, days=1
            )

            # Calculate feature statistics
            features = {}

            # Quantiles
            features["mc_1d_q1"] = np.percentile(simulated_prices, 1)
            features["mc_1d_q5"] = np.percentile(simulated_prices, 5)
            features["mc_1d_q10"] = np.percentile(simulated_prices, 10)
            features["mc_1d_q50"] = np.percentile(simulated_prices, 50)
            features["mc_1d_q95"] = np.percentile(simulated_prices, 95)

            # Expected value and return
            expected_value = simulated_prices.mean()
            features["mc_1d_expected_value"] = expected_value
            features["mc_1d_return_pct"] = (
                (expected_value - current_price) / current_price
            ) * 100

            # Probability of breakeven (positive return)
            features["mc_1d_prob_breakeven"] = np.mean(simulated_prices > current_price)

            return features

        except Exception as e:
            # Return None on any error - will be handled gracefully
            return None

    def add_all(self, df):
        """
        Add Monte Carlo features to DataFrame using vectorized simulation.

        Parameters:
            df (pd.DataFrame): DataFrame with price data

        Returns:
            pd.DataFrame: DataFrame with MC features added
        """
        df_modified = df.copy()

        # Initialize feature columns with NaN
        for feature_name in self.features:
            df_modified[feature_name] = np.nan
        
        if "mc_1d_expected_value" not in df_modified.columns:
            df_modified["mc_1d_expected_value"] = np.nan

        if len(df_modified) < self.min_data_threshold:
            return df_modified

        # 1. Calculate log returns
        log_returns = np.log(df_modified[self.price_col] / df_modified[self.price_col].shift(1))

        # 2. Pre-calculate rolling drift and volatility
        # Using the same window logic as before
        rolling_mean = log_returns.rolling(window=self.window_size).mean()
        rolling_var = log_returns.rolling(window=self.window_size).var()
        rolling_std = log_returns.rolling(window=self.window_size).std()

        drifts = (rolling_mean - 0.5 * rolling_var).values
        vols = rolling_std.values
        current_prices = df_modified[self.price_col].values

        # 3. Vectorized simulation for each valid row
        # Set random seed
        np.random.seed(self.random_seed)
        dt = 1 / 252
        
        # To handle all rows efficiently, we'll iterate through valid indices
        # but the simulation itself is vectorized over iterations.
        # Fully vectorizing across ALL rows AND ALL iterations might consume too much memory
        # (e.g., 1000 rows * 1000 iterations = 1M values, which is fine, but let's be careful).
        
        valid_indices = range(self.min_data_threshold, len(df_modified))
        
        # Pre-generate shocks for all simulations to be fast
        # (days_to_simulate=1, iterations, num_valid_rows)
        all_shocks = np.random.normal(size=(self.iterations, len(valid_indices)))
        
        # Get parameters for valid rows only using iloc/positional access to numpy arrays
        valid_drifts = drifts[valid_indices]
        valid_vols = vols[valid_indices]
        valid_prices = current_prices[valid_indices]
        
        # Reshape for broadcasting: (iterations, num_valid_rows)
        # drift/vol/prices are (num_valid_rows,) -> (1, num_valid_rows)
        valid_drifts = valid_drifts.reshape(1, -1)
        valid_vols = valid_vols.reshape(1, -1)
        valid_prices = valid_prices.reshape(1, -1)
        
        # Calculate end prices: (iterations, num_valid_rows)
        daily_returns = np.exp(valid_drifts * dt + valid_vols * np.sqrt(dt) * all_shocks)
        simulated_prices = valid_prices * daily_returns
        
        # 4. Calculate statistics across iterations (axis=0)
        # Statistics will be (num_valid_rows,)
        q1 = np.percentile(simulated_prices, 1, axis=0)
        q5 = np.percentile(simulated_prices, 5, axis=0)
        q10 = np.percentile(simulated_prices, 10, axis=0)
        q50 = np.percentile(simulated_prices, 50, axis=0)
        q95 = np.percentile(simulated_prices, 95, axis=0)
        expected_values = np.mean(simulated_prices, axis=0)
        prob_breakeven = np.mean(simulated_prices > valid_prices, axis=0)
        returns_pct = ((expected_values - valid_prices.flatten()) / valid_prices.flatten()) * 100
        
        # 5. Assign back to DataFrame using labels from the index to ensure alignment
        idx_array = df_modified.index[valid_indices]
        
        if "mc_1d_q1" in df_modified.columns: df_modified.loc[idx_array, "mc_1d_q1"] = q1
        if "mc_1d_q5" in df_modified.columns: df_modified.loc[idx_array, "mc_1d_q5"] = q5
        if "mc_1d_q10" in df_modified.columns: df_modified.loc[idx_array, "mc_1d_q10"] = q10
        if "mc_1d_q50" in df_modified.columns: df_modified.loc[idx_array, "mc_1d_q50"] = q50
        if "mc_1d_q95" in df_modified.columns: df_modified.loc[idx_array, "mc_1d_q95"] = q95
        if "mc_1d_return_pct" in df_modified.columns: df_modified.loc[idx_array, "mc_1d_return_pct"] = returns_pct
        if "mc_1d_prob_breakeven" in df_modified.columns: df_modified.loc[idx_array, "mc_1d_prob_breakeven"] = prob_breakeven
        df_modified.loc[idx_array, "mc_1d_expected_value"] = expected_values

        return df_modified

    def add_mc_features(self, df, price_col=None):
        """
        Backward-compatible wrapper for older scripts.
        """
        original_price_col = self.price_col
        if price_col is not None:
            self.price_col = price_col
        try:
            return self.add_all(df)
        finally:
            self.price_col = original_price_col

    def validate_features(self, df):
        """
        Validate MC features for data quality.

        Parameters:
            df (pd.DataFrame): DataFrame with MC features

        Returns:
            dict: Validation report
        """
        report = {"total_rows": len(df), "valid_rows": 0, "nan_rows": 0, "errors": []}

        # Count valid rows
        mc_cols = [col for col in df.columns if col.startswith("mc_1d_")]
        if not mc_cols:
            report["errors"].append("No Monte Carlo feature columns found")
            report["nan_rows"] = len(df)
            return report

        valid_mask = df[mc_cols].notna().all(axis=1)
        report["valid_rows"] = valid_mask.sum()
        report["nan_rows"] = len(df) - report["valid_rows"]

        # Check quantile ordering
        if "mc_1d_q1" in df.columns and "mc_1d_q95" in df.columns:
            valid_data = df[valid_mask]
            ordering_violations = (
                (valid_data["mc_1d_q1"] >= valid_data["mc_1d_q5"])
                | (valid_data["mc_1d_q5"] >= valid_data["mc_1d_q10"])
                | (valid_data["mc_1d_q10"] >= valid_data["mc_1d_q50"])
                | (valid_data["mc_1d_q50"] >= valid_data["mc_1d_q95"])
            ).sum()

            if ordering_violations > 0:
                report["errors"].append(
                    f"Quantile ordering violations: {ordering_violations}"
                )

        # Check probability bounds
        if "mc_1d_prob_breakeven" in df.columns:
            valid_data = df[valid_mask]
            prob_violations = (
                (valid_data["mc_1d_prob_breakeven"] < 0)
                | (valid_data["mc_1d_prob_breakeven"] > 1)
            ).sum()

            if prob_violations > 0:
                report["errors"].append(f"Probability out of bounds: {prob_violations}")

        # Check for infinite values
        inf_count = np.isinf(df[mc_cols]).sum().sum()
        if inf_count > 0:
            report["errors"].append(f"Infinite values detected: {inf_count}")

        return report

    @staticmethod
    def get_feature_names(horizons=None):
        """
        Return list of MC feature names for given forecast horizons.

        Parameters:
            horizons (list, optional): List of day horizons to generate
                feature names for (e.g. [1, 3, 7]). Defaults to [1].

        Returns:
            list: Feature names in the format mc_{n}d_{statistic}
        """

        if horizons is None:
            horizons = [1]

        # statistic suffixes generated per horizon
        suffixes = [
            "q1",
            "q5",
            "q10",
            "q50",
            "q95",
            "return_pct",
            "prob_breakeven",
        ]

        feature_names = []
        for horizon in horizons:
            for suffix in suffixes:
                feature_names.append(f"mc_{horizon}d_{suffix}")

        return feature_names

    def print_feature_summary(self, df):
        """
        Print summary statistics of MC features.

        Parameters:
            df (pd.DataFrame): DataFrame with MC features
        """
        print("\nMC Feature Statistics:")
        print("-" * 60)

        mc_cols = [col for col in df.columns if col.startswith("mc_1d_")]

        for col in mc_cols:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                print(
                    f"{col:30s}: [{valid_data.min():8.2f}, {valid_data.max():8.2f}], "
                    f"mean={valid_data.mean():8.2f}"
                )
            else:
                print(f"{col:30s}: No valid data")
