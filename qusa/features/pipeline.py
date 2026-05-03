# QUSA/qusa/features/pipeline.py

from qusa.features.overnight import OvernightCalculator
from qusa.features.calendar import CalendarFeatures
from qusa.features.technical import TechnicalIndicators
from qusa.features.monte_carlo import MonteCarloFeatures


class FeaturePipeline:
    """
    Pipeline to apply multiple feature calculations to financial time series data.
    """

    def __init__(self, config=None):
        """
        Class constructor.

        Parameters:
            1) config (dict): Configuration dictionary with pipeline settings.
        """

        self.config = config or {}

        self.overnight_calculator = OvernightCalculator(
            date_col=self.config.get("date_col", "date"),
            open_col=self.config.get("open_col", "open"),
            close_col=self.config.get("close_col", "close"),
        )
        self.calendar_features = CalendarFeatures(
            date_col=self.config.get("date_col", "date")
        )
        self.technical_indicators = TechnicalIndicators(
            config=self.config.get("technical_params", {}),
            date_col=self.config.get("date_col", "date"),
            open_col=self.config.get("open_col", "open"),
            close_col=self.config.get("close_col", "close"),
            high_col=self.config.get("high_col", "high"),
            low_col=self.config.get("low_col", "low"),
            volume_col=self.config.get("volume_col", "volume"),
        )

        # Initialize Monte Carlo features
        mc_config = self.config.get("monte_carlo", {})
        if mc_config.get("enabled", False):
            mc_config = {
                **mc_config,
                "price_col": self.config.get("close_col", "close"),
            }
            self.monte_carlo = MonteCarloFeatures(config=mc_config)
        else:
            self.monte_carlo = None

    def run(self, df, ticker=None):
        """
        Run the feature pipeline on the provided DataFrame.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data.

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with all features added.
        """

        df_mod = df.copy()

        # Step 1) Calculate overnight features
        df_mod = self.overnight_calculator.calculate_overnight_delta(df_mod)
        df_mod = self.overnight_calculator.identify_abnormal_delta(
            df_mod,
            threshold=self.config.get("overnight", {}).get("abnormal_threshold", 2.0),
        )

        # Step 2) Calculate technical indicators
        df_mod = self.technical_indicators.add_all(df_mod)

        # Step 3) Calculate calendar features
        df_mod = self.calendar_features.add_all(df_mod)

        # Step 4) Calculate Monte Carlo features (if enabled)
        if self.monte_carlo is not None:
            df_mod = self.monte_carlo.add_all(df_mod)

        return df_mod

    @staticmethod
    def get_engineered_features(include_monte_carlo=True, mc_horizons=None):
        """
        Get the list of all engineered feature names.

        Returns:
            1) features (list): List of engineered feature names.
        """

        features = []

        # Overnight features
        features.extend(
            ["overnight_delta", "overnight_delta_pct", "abnormal_overnight_delta"]
        )

        # Technical indicators
        features.extend(
            [
                "rsi",
                "atr",
                "volume_ma",
                "52_week_high_proximity",
                "52_week_low_proximity",
            ]
        )

        # Calendar features
        features.extend(
            [
                "day_of_week",
                "is_monday",
                "month_of_year",
                "is_jan",
                "is_month_start",
                "is_month_end",
            ]
        )

        # Monte Carlo features
        if include_monte_carlo:
            features.extend(MonteCarloFeatures.get_feature_names(horizons=mc_horizons))

        return features
