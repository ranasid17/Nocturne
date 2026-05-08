# QUSA/qusa/features/calendar.py


import pandas as pd


class CalendarFeatures:
    """
    Calculates calendar-based features for financial time series data.
    """

    def __init__(self, date_col="date"):
        """
        Class constructor.
        Parameters:
            1) date_col (str): Name of the date column.
        """

        self.date = date_col

    def add_all(self, df):
        """
        Label all calendar features.

        Parameters:
            1) df (pd.DataFrame): DataFrame with stock data

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with all calendar features added
        """

        df_mod = self.label_day_of_week(df)
        df_mod = self.label_month_of_year(df_mod)
        df_mod = self.label_start_end_of_month(df_mod)

        return df_mod

    def label_day_of_week(self, df):
        """
        Label day of week and one-hot encode.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing 'date' column

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with days of week labeled
        """

        # copy the original DataFrame to avoid direct modification
        df_mod = df.copy()

        # label day of week
        df_mod["day_of_week"] = df_mod[self.date].dt.day_of_week

        # one-hot encode day of week
        df_mod["is_monday"] = df_mod["day_of_week"] == 0
        df_mod["is_tuesday"] = df_mod["day_of_week"] == 1
        df_mod["is_wednesday"] = df_mod["day_of_week"] == 2
        df_mod["is_thursday"] = df_mod["day_of_week"] == 3
        df_mod["is_friday"] = df_mod["day_of_week"] == 4

        return df_mod

    def label_month_of_year(self, df):
        """
        Label month of year and one-hot encode.

        Parameters:
            1) df (pd.DataFrame): DataFrame with 'date' column

        Returns:
            1) df (pd.DataFrame): DataFrame with months labeled
        """

        # copy the original DataFrame to avoid direct modification
        df_mod = df.copy()

        # label months
        df_mod["month_of_year"] = df_mod[self.date].dt.month

        # one hot encode month of year
        df_mod["is_jan"] = df_mod["month_of_year"] == 1
        df_mod["is_feb"] = df_mod["month_of_year"] == 2
        df_mod["is_mar"] = df_mod["month_of_year"] == 3
        df_mod["is_apr"] = df_mod["month_of_year"] == 4
        df_mod["is_may"] = df_mod["month_of_year"] == 5
        df_mod["is_jun"] = df_mod["month_of_year"] == 6
        df_mod["is_jul"] = df_mod["month_of_year"] == 7
        df_mod["is_aug"] = df_mod["month_of_year"] == 8
        df_mod["is_sep"] = df_mod["month_of_year"] == 9
        df_mod["is_oct"] = df_mod["month_of_year"] == 10
        df_mod["is_nov"] = df_mod["month_of_year"] == 11
        df_mod["is_dec"] = df_mod["month_of_year"] == 12

        return df_mod

    def label_start_end_of_month(self, df):
        """
        Label first, final 5 days of each month.

        Parameters:
            1) df (pd.DataFrame): DataFrame with 'date' column

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with start, end months labeled
        """

        # copy the original DataFrame to avoid direct modification
        df_mod = df.copy()

        # extract date within month
        df_mod["day_of_month"] = df_mod[self.date].dt.day

        # label first, final 5d of month
        df_mod["first_5d_month"] = df_mod["day_of_month"] <= 5
        df_mod["final_5d_month"] = df_mod["day_of_month"] >= 25

        return df_mod
