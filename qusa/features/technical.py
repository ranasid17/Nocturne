# QUSA/qusa/features/technical.py

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    Calculates technical indicators for financial time series data.
    """

    def __init__(
        self,
        config,
        open_col="open",
        date_col="date",
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
    ):
        """
        Class constructor.

        Parameters:
            1) config (dict): Configuration dictionary.
            1) date_col (str): Name of the date column.
            2) open_col (str): Name of the opening price column.
            3) close_col (str): Name of the closing price column.
            4) high_col (str): Name of the high price column.
            5) low_col (str): Name of the low price column.
            6) volume_col (str): Name of the volume column.

        """

        self.config = config or {}

        # periods from config with defaults if not found/specified
        self.period_rsi = self.config.get("rsi_window", 14)
        self.period_atr = self.config.get("atr_window", 14)
        self.period_volume = self.config.get("volume_ma_window", 20)
        self.period_rolling = self.config.get("rolling_window_52w", 252)

        # column names
        self.date = date_col
        self.open = open_col
        self.close = close_col
        self.high = high_col
        self.low = low_col
        self.volume = volume_col

    def add_all(self, df):
        """
        Calculate all technical indicators.

        Parameters:
            1) df (pd.DataFrame): DataFrame with stock data

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with all technical indicators added
        """

        df_mod = df.copy()  # copy input DataFrame to avoid direct modification

        # calculate individual technical indicators
        df_mod = self.calculate_volume_spike(df_mod)
        df_mod = self.calculate_rsi(df_mod)
        df_mod = self.calculate_average_true_range(df_mod)
        df_mod = self.calculate_annual_min_max_proximity(df_mod)
        df_mod = self.calculate_intraday_momentum(df_mod)
        df_mod = self.calculate_late_day_momentum(df_mod)

        return df_mod

    def calculate_volume_spike(self, df, threshold=2.0):
        """
        Calculate volume spike.

        Parameters:
            1) df (pd.DataFrame): DataFrame with stock data and 'volume' column
            2) threshold (float): Threshold ratio for spike detection. Default is 2.

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with new indicator column
        """

        # copy input DataFrame to avoid direct modification
        df_mod = df.copy()

        # calculate 20d moving average
        df_mod["avg_volume_20"] = (
            df_mod["volume"].rolling(window=self.period_volume).mean()
        )
        # divide volume count by mean
        df_mod["volume_ratio"] = df_mod["volume"] / df_mod["avg_volume_20"]
        # label spikes greater than input threshold
        df_mod["volume_spike"] = df_mod["volume_ratio"] > threshold

        return df_mod

    def calculate_rsi(self, df):
        """
        Calculate relative strength indicator (RSI) using Wilder's EMA.

        Parameters:
            1) df (pd.DataFrame): DataFrame with stock data and 'close' column

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with additional technical indicator column
        """

        # copy input DataFrame to avoid direct modification
        df_mod = df.copy()

        # calculate daily price change (close, close)
        delta = df_mod[self.close].diff()

        # calculate positive, negative price changes
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))

        # Wilder's Smoothing (EMA with alpha = 1/period)
        avg_gain = gain.ewm(com=self.period_rsi - 1, min_periods=self.period_rsi).mean()
        avg_loss = loss.ewm(com=self.period_rsi - 1, min_periods=self.period_rsi).mean()

        # calculate relative strength and add to DataFrame
        relative_strength = avg_gain / avg_loss

        df_mod["rsi"] = 100 - (100 / (1 + relative_strength))
        df_mod["rsi_oversold"] = df_mod["rsi"] < 30
        df_mod["rsi_overbought"] = df_mod["rsi"] > 70

        return df_mod

    def calculate_average_true_range(self, df):
        """
        Calculate average true range (ATR).

        Parameters:
            1) df (pd.DataFrame): DataFrame with 'high', 'low, 'close' columns
            2) period (int): Period for ATR calculation. Default is 14.

        Returns:
            1) df_mod (pd.DataFrame): Dataframe with new technical indicator column
        """

        # copy input DataFrame to avoid direct modification
        df_mod = df.copy()

        # calculate intraday max (high, low) range
        price_range_high_low = df_mod[self.high] - df_mod[self.low]
        # calculate intraday (high, close) range
        price_range_high_close = np.abs(df_mod[self.high] - df_mod[self.close].shift())
        # calculate intraday (low, close) range
        price_range_low_close = np.abs(df_mod[self.low] - df_mod[self.close].shift())

        # concatenate intraday ranges into Series object
        price_ranges = pd.concat(
            [price_range_high_low, price_range_high_close, price_range_low_close],
            axis=1,
        )
        # store max range per day
        daily_max_range = np.max(price_ranges, axis=1)

        # calculate 14d rolling max price change (standard ATR uses smoothing, but keeping as is for now or switching to EMA if desired)
        df_mod["atr"] = daily_max_range.rolling(window=self.period_atr).mean()
        df_mod["atr_pct"] = (df_mod["atr"] / df_mod[self.close]) * 100

        return df_mod

    def calculate_annual_min_max_proximity(self, df):
        """
        Calculate daily proximity to 52 wk high/low.

        Parameters:
            1) df (pd.DataFrame): Contains 'high', 'low' columns
            2) period (int): Period for rolling min, max calculation. Default is 252.

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with new technical indicator
        """

        # copy input DataFrame to avoid direct modification
        df_mod = df.copy()

        # store rolling min, max values
        df_mod["52_week_high"] = (
            df_mod[self.high].rolling(window=self.period_rolling).max()
        )
        df_mod["52_week_low"] = (
            df_mod[self.low].rolling(window=self.period_rolling).min()
        )

        # calculate relative value from daily close to 52-week high, low
        df_mod["52_week_high_proximity"] = (
            (df_mod["52_week_high"] - df_mod[self.close]) / df_mod["52_week_high"]
        ) * 100
        df_mod["52_week_low_proximity"] = (
            (df_mod[self.close] - df_mod["52_week_low"]) / df_mod["52_week_low"]
        ) * 100

        # label when within threshold (5%) of 52 wk high, low
        df_mod["52_week_high_threshold"] = df_mod["52_week_high_proximity"] < 5
        df_mod["52_week_low_threshold"] = df_mod["52_week_low_proximity"] < 5

        return df_mod

    def calculate_intraday_momentum(self, df, threshold=2.0):
        """
        Calculate intraday momentum.

        Parameters:
            1) df (pd.DataFrame): DataFrame with 'open', 'close' columns
            2) threshold (float): Threshold percentage for strong momentum. Default is 2.0.

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with new technical indicator
        """

        # copy input DataFrame to avoid direct modification
        df_mod = df.copy()

        # calculate relative intraday price change
        df_mod["intraday_return"] = (
            (df_mod[self.close] - df_mod[self.open]) / df_mod[self.open]
        ) * 100

        # label strong, weak (+/- 2%) relative intraday price change
        df_mod["intraday_return_strong_positive"] = (
            df_mod["intraday_return"] > threshold
        )
        df_mod["intraday_return_strong_negative"] = (
            df_mod["intraday_return"] < -threshold
        )

        return df_mod

    def calculate_late_day_momentum(self, df):
        """
        Calculate late day momentum (close position relative to daily range).

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data with 'close' column

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with new indicator column
        """

        df_mod = df.copy()  # copy input DataFrame to avoid direct modification

        # 1) late day momentum - standard close position: (close - low) / (high - low)
        # handle division by zero where high == low
        range_hl = (df_mod[self.high] - df_mod[self.low])
        df_mod["close_position"] = (df_mod[self.close] - df_mod[self.low]) / range_hl.replace(0, np.nan)
        df_mod["close_position"] = df_mod["close_position"].fillna(0.5)

        return df_mod
