# QUSA/qusa/features/pipeline.py

from qusa.features.overnight import OvernightCalculator
from qusa.features.calendar import CalendarFeatures
from qusa.features.technical import TechnicalIndicators
from qusa.features.monte_carlo import MonteCarloFeatures
from qusa.features.volatility import VolatilityCalculator


MODEL_FEATURE_MANIFEST_VERSION = "v2"
MODEL_BASE_FEATURES = [
    "52_week_high_proximity",
    "52_week_low_proximity",
    "atr_pct",
    "close_position",
    "rsi",
    "volume_ratio",
    "day_of_week",
    "day_of_month",
    "month_of_year",
    "first_5d_month",
    "final_5d_month",
    "is_monday",
    "is_tuesday",
    "is_wednesday",
    "is_thursday",
    "is_friday",
    "is_jan",
    "is_feb",
    "is_mar",
    "is_apr",
    "is_may",
    "is_jun",
    "is_jul",
    "is_aug",
    "is_sep",
    "is_oct",
    "is_nov",
    "is_dec",
    "vwap_deviation",
    "vol_regime",
]


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
        feature_params = self.config.get("feature_params")
        if feature_params is None:
            feature_params = self.config.get("technical_params", {})
        volatility_params = self.config.get("feature_params")
        if volatility_params is None:
            volatility_params = self.config.get("features", feature_params)

        self.overnight_calculator = OvernightCalculator(
            date_col=self.config.get("date_col", "date"),
            open_col=self.config.get("open_col", "open"),
            close_col=self.config.get("close_col", "close"),
        )
        self.calendar_features = CalendarFeatures(
            date_col=self.config.get("date_col", "date")
        )
        self.volatility_calculator = VolatilityCalculator(
            config=volatility_params
        )
        self.technical_indicators = TechnicalIndicators(
            config=feature_params,
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

        # Step 3) Calculate volatility features (Task 5.1)
        df_mod = self.volatility_calculator.add_all(df_mod)

        # Step 4) Calculate calendar features
        df_mod = self.calendar_features.add_all(df_mod)

        # Step 5) Calculate Monte Carlo features (if enabled)
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

        # Volatility features (Task 5.1)
        features.extend(["vwap_deviation", "vol_regime", "vol_parkinson", "vol_garman_klass"])

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

    @staticmethod
    def get_model_feature_manifest(include_monte_carlo=False, mc_horizons=None):
        """Return the ordered feature contract used by model training and inference."""

        features = MODEL_BASE_FEATURES.copy()
        if include_monte_carlo:
            features.extend(MonteCarloFeatures.get_feature_names(horizons=mc_horizons))
        return list(dict.fromkeys(features))
