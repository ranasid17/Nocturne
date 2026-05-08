# qusa/features/monte_carlo.py

"""
Monte Carlo feature generation for stock price forecasting.
Uses geometric Brownian motion to simulate future price paths.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


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
        self.horizons = self.config.get("horizons", [1])
        
        # All potential features across all horizons
        self.features = self.get_feature_names(horizons=self.horizons)

    def calculate_log_returns(self, prices):
        """
        Calculate log returns from price series.

        Parameters:
            prices (pd.Series): Price series

        Returns:
            pd.Series: Log returns
        """
        return np.log(prices / prices.shift(1))

    def add_all(self, df):
        """
        Add Monte Carlo features to DataFrame using vectorized simulation.
        Supports multi-horizon forecasting and incremental checkpointing.

        Parameters:
            df (pd.DataFrame): DataFrame with price data

        Returns:
            pd.DataFrame: DataFrame with MC features added
        """
        df_modified = df.copy()

        # 1. Check for incremental compute (Task 5.3)
        # Identify rows where any required MC feature is NaN
        required_cols = self.features
        
        # Initialize missing columns
        for col in required_cols:
            if col not in df_modified.columns:
                df_modified[col] = np.nan
        
        if "mc_1d_expected_value" not in df_modified.columns:
            df_modified["mc_1d_expected_value"] = np.nan

        if len(df_modified) < self.min_data_threshold:
            return df_modified

        # Determine indices to compute
        # We compute for rows after threshold where features are missing
        potential_indices = np.arange(self.min_data_threshold, len(df_modified))
        
        # For incremental: check if last row is already computed
        # Actually, if any MC col is NaN in valid range, we compute.
        # This is simpler and robust to partial failures.
        missing_mask = df_modified[required_cols].isna().any(axis=1)
        valid_indices = [i for i in potential_indices if missing_mask.iloc[i]]

        if not valid_indices:
            logger.info("MC features already computed, skipping.")
            return df_modified

        logger.info(f"Computing MC features for {len(valid_indices)} rows...")

        # 2. Pre-calculate log returns and rolling stats
        log_returns = np.log(df_modified[self.price_col] / df_modified[self.price_col].shift(1))
        rolling_mean = log_returns.rolling(window=self.window_size).mean()
        rolling_var = log_returns.rolling(window=self.window_size).var()
        rolling_std = log_returns.rolling(window=self.window_size).std()

        drifts = (rolling_mean - 0.5 * rolling_var).values
        vols = rolling_std.values
        current_prices = df_modified[self.price_col].values

        # Set random seed
        np.random.seed(self.random_seed)
        dt = 1 / 252
        
        # Get parameters for rows to compute
        comp_drifts = drifts[valid_indices].reshape(1, -1)
        comp_vols = vols[valid_indices].reshape(1, -1)
        comp_prices = current_prices[valid_indices].reshape(1, -1)
        
        # 3. Iterate over horizons (Task 5.2)
        for h in self.horizons:
            # Pre-generate shocks: (h, iterations, num_comp_rows)
            # For 1-day, h=1. 
            # Vectorizing over iterations and rows.
            # Total values = h * iterations * len(valid_indices)
            # If h=1, iterations=1000, rows=1000 -> 1M values (OK)
            
            # Use sum of shocks for cumulative returns in GBM
            # log(S_T/S_0) ~ Normal((mu - 0.5*sigma^2)*T, sigma*sqrt(T)*Z)
            # where Z ~ N(0, 1) and T = h * dt
            
            shocks = np.random.normal(size=(self.iterations, len(valid_indices)))
            
            T = h * dt
            drift_term = comp_drifts * T
            vol_term = comp_vols * np.sqrt(T) * shocks
            
            simulated_prices = comp_prices * np.exp(drift_term + vol_term)
            
            # 4. Calculate statistics
            q1 = np.percentile(simulated_prices, 1, axis=0)
            q5 = np.percentile(simulated_prices, 5, axis=0)
            q10 = np.percentile(simulated_prices, 10, axis=0)
            q50 = np.percentile(simulated_prices, 50, axis=0)
            q95 = np.percentile(simulated_prices, 95, axis=0)
            expected_values = np.mean(simulated_prices, axis=0)
            prob_breakeven = np.mean(simulated_prices > comp_prices, axis=0)
            returns_pct = ((expected_values - comp_prices.flatten()) / comp_prices.flatten()) * 100
            
            # 5. Assign back to DataFrame
            idx_array = df_modified.index[valid_indices]
            
            df_modified.loc[idx_array, f"mc_{h}d_q1"] = q1
            df_modified.loc[idx_array, f"mc_{h}d_q5"] = q5
            df_modified.loc[idx_array, f"mc_{h}d_q10"] = q10
            df_modified.loc[idx_array, f"mc_{h}d_q50"] = q50
            df_modified.loc[idx_array, f"mc_{h}d_q95"] = q95
            df_modified.loc[idx_array, f"mc_{h}d_return_pct"] = returns_pct
            df_modified.loc[idx_array, f"mc_{h}d_prob_breakeven"] = prob_breakeven
            
            if h == 1:
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

    def get_feature_summary_string(self, df):
        """
        Return a summary string of MC features.
        """
        summary = "MC Feature Statistics:\n"
        summary += "-" * 60 + "\n"

        for h in self.horizons:
            prefix = f"mc_{h}d_"
            mc_cols = [col for col in df.columns if col.startswith(prefix)]

            for col in mc_cols:
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    summary += (
                        f"{col:30s}: [{valid_data.min():8.2f}, {valid_data.max():8.2f}], "
                        f"mean={valid_data.mean():8.2f}\n"
                    )
                else:
                    summary += f"{col:30s}: No valid data\n"
        return summary

    def print_feature_summary(self, df):
        """
        Print summary statistics of MC features.

        Parameters:
            df (pd.DataFrame): DataFrame with MC features
        """
        print("\nMC Feature Statistics:")
        print("-" * 60)

        # Print for all horizons
        for h in self.horizons:
            prefix = f"mc_{h}d_"
            mc_cols = [col for col in df.columns if col.startswith(prefix)]

            for col in mc_cols:
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    print(
                        f"{col:30s}: [{valid_data.min():8.2f}, {valid_data.max():8.2f}], "
                        f"mean={valid_data.mean():8.2f}"
                    )
                else:
                    print(f"{col:30s}: No valid data")
