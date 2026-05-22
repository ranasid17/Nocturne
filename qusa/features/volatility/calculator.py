# qusa/features/volatility/calculator.py

import pandas as pd
import numpy as np


class VolatilityCalculator:
    """
    Calculates volatility-related features.
    """

    def __init__(self, config=None):
        """
        Class constructor.
        """
        self.config = config or {}
        self.vwap_window = self.config.get("vwap_window", 20)
        self.vol_regime_short = self.config.get("vol_regime_short_window", 5)
        self.vol_regime_long = self.config.get("vol_regime_long_window", 20)
        self.advanced_vol_window = self.config.get("advanced_vol_window", 20)

    def calculate_vwap_deviation(self, df):
        """
        Calculate price deviation from Volume Weighted Average Price (VWAP).
        """
        df_mod = df.copy()
        
        # Typical price * volume
        # Actually use close for simplicity per plan: (close - vwap) / vwap
        tp_v = df_mod["close"] * df_mod["volume"]
        
        rolling_tp_v = tp_v.rolling(window=self.vwap_window).sum()
        rolling_v = df_mod["volume"].rolling(window=self.vwap_window).sum()
        
        vwap = rolling_tp_v / rolling_v
        
        df_mod["vwap_deviation"] = (df_mod["close"] - vwap) / vwap * 100
        return df_mod

    def calculate_vol_regime(self, df):
        """
        Calculate relative volatility regime.
        """
        df_mod = df.copy()
        
        log_returns = np.log(df_mod["close"] / df_mod["close"].shift(1))
        
        vol_short = log_returns.rolling(window=self.vol_regime_short).std()
        vol_long = log_returns.rolling(window=self.vol_regime_long).std()
        
        df_mod["vol_regime"] = vol_short / vol_long
        return df_mod

    def calculate_advanced_vol(self, df, window=20):
        """
        Calculate Parkinson and Garman-Klass volatility.
        """
        df_mod = df.copy()
        
        # Parkinson Volatility
        pk_val = (1.0 / (4.0 * np.log(2.0))) * np.square(np.log(df_mod["high"] / df_mod["low"]))
        df_mod["vol_parkinson"] = np.sqrt(pk_val.rolling(window=window).mean())
        
        # Garman-Klass Volatility
        log_hl = np.square(np.log(df_mod["high"] / df_mod["low"]))
        log_co = np.square(np.log(df_mod["close"] / df_mod["open"]))
        
        gk_val = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
        df_mod["vol_garman_klass"] = np.sqrt(gk_val.rolling(window=window).mean())
        
        return df_mod

    def add_all(self, df):
        """
        Add all volatility features.
        """
        df_mod = self.calculate_vwap_deviation(df)
        df_mod = self.calculate_vol_regime(df_mod)
        df_mod = self.calculate_advanced_vol(df_mod, window=self.advanced_vol_window)
        return df_mod
