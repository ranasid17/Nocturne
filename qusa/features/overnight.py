# QUSA/qusa/features/overnight.py

import pandas as pd


class OvernightCalculator:
    """
    Calculates overnight features for financial time series data.
    """

    def __init__(self, date_col="date", open_col="open", close_col="close"):
        """
        Class constructor.

        Parameters:
            1) date_col (str): Name of the date column.
            2) open_col (str): Name of the opening price column.
            3) close_col (str): Name of the closing price column.
        """

        self.date = date_col
        self.open = open_col
        self.close = close_col

    def calculate_overnight_delta(self, df):
        """
        Calculate the overnight price change for each trading day in the DataFrame.

        Parametesrs:
            1) df (pd.DataFrame): DataFrame containing stock data with 'Date', 'Open', and 'Close' columns.

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with an additional 'Overnight_Delta' column.
        """

        df_mod = df.copy()  # copy the original DataFrame to avoid modifying it directly

        df_mod[self.date] = pd.to_datetime(  # confirm date column as datetime type
            df_mod[self.date]
        )
        df_mod = df_mod.sort_values(by=self.date).reset_index(
            drop=True
        )  # sort by date and reset index

        # calculate overnight change and percentage change
        df_mod["overnight_delta"] = df_mod[self.open] - df_mod[self.close].shift(1)
        df_mod["overnight_delta_pct"] = (
            df_mod["overnight_delta"] / df_mod[self.close].shift(1)
        ) * 100

        return df_mod

    @staticmethod
    def identify_abnormal_delta(df, threshold=2.0, window=252):
        """
        Identify abnormal overnight price changes using rolling statistics.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data with 'Overnight_Delta' column.
            2) threshold (float): Threshold value for identifying abnormal overnight delta.
            3) window (int): Rolling window size for mean/std calculation. Default is 252.

        Returns:
            1) df_mod (pd.DataFrame): DataFrame containing days with abnormal overnight delta.
        """

        df_mod = df.copy()  # copy the original DataFrame to avoid modifying it directly

        # calculate rolling mean and standard deviation to avoid look-ahead bias
        # we require at least 30 observations for valid stats
        rolling_mean = df_mod["overnight_delta_pct"].rolling(window=window, min_periods=30).mean()
        rolling_std = df_mod["overnight_delta_pct"].rolling(window=window, min_periods=30).std()

        # calculate z score and label anomalies using rolling stats
        df_mod["z_score"] = (df_mod["overnight_delta_pct"] - rolling_mean) / rolling_std
        df_mod["abnormal"] = df_mod["z_score"].abs() > threshold

        return df_mod
