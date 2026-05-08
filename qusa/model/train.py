# qusa/qusa/model/train.py

"""
Train model to predict overnight price direction.
"""

import logging
import joblib
import os
import pandas as pd

from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

# define allowed features for training
SAFE_FEATURES = [
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
    # Volatility features
    "vwap_deviation",
    "vol_regime",
    # Monte Carlo features
    "mc_1d_q1",
    "mc_1d_q5",
    "mc_1d_q10",
    "mc_1d_q50",
    "mc_1d_q95",
    "mc_1d_return_pct",
    "mc_1d_prob_breakeven",
]

# confirm no duplicate features
SAFE_FEATURES = list(dict.fromkeys(SAFE_FEATURES))


def get_safe_features(include_monte_carlo=True, mc_horizons=None):
    """
    Return SAFE_FEATURES plus Monte Carlo feature names for any horizons
    beyond those already included in the base SAFE_FEATURES list.

    Parameters:
        include_monte_carlo (bool): Whether to include MC features.
        mc_horizons (list, optional): Forecast horizons in days. Defaults
            to [1] inside MonteCarloFeatures.get_feature_names().

    Returns:
        list: Deduplicated list of safe feature names.
    """
    features = SAFE_FEATURES.copy()

    if include_monte_carlo:
        try:
            from qusa.features.monte_carlo import MonteCarloFeatures

            mc_feature_names = MonteCarloFeatures.get_feature_names(
                horizons=mc_horizons
            )
            features.extend(mc_feature_names)
        except Exception:
            # If import fails, return base features only
            pass

    # deduplicate while preserving order
    return list(dict.fromkeys(features))


def prepare_model_features(data, feature_names):
    """
    Select model features and replace missing or non-finite values.
    """

    return data[feature_names].replace([float("inf"), -float("inf")], 0).fillna(0)


# define leakage features
CONFOUND_FEATURES = [
    "overnight_delta",  # target feature
    "overnight_delta_pct",  # target feature
    "date",  # not a feature
    "z_score",  # derived from target
    "abnormal",  # derived from target
    "intraday_returns",  # calculated next day
    "intraday_return_strong_positive",  # calculated next day
    "intraday_return_strong_negative",  # calculated next day
]


class OvernightDirectionModel:
    """
    Decision tree model to predict overnight price movement.
    """

    def __init__(self, config=None):
        """
        Class constructor.

        Parameters:
            1) config (dict): Model configuration
        """

        self.config = config
        self.model = None
        self.feature_names = SAFE_FEATURES
        self.trained_date = None
        self.metrics = {}

        # determine whether to include Monte Carlo features from config (safe default False)
        include_mc = False
        mc_horizons = None
        if isinstance(config, dict):
            mc_conf = config.get("monte_carlo", {})
            include_mc = mc_conf.get("enabled", False)
            mc_horizons = mc_conf.get("horizons", None)

        # set feature names at runtime using lazy importer
        self.feature_names = get_safe_features(
            include_monte_carlo=include_mc, mc_horizons=mc_horizons
        )

    @staticmethod
    def load_data(data_path):
        """
        Load and prepare data

        Parameters:
            1) data_path (str): Path to data for model
        """

        logger.info("Loading data...")
        data = pd.read_csv(os.path.expanduser(data_path))

        ###
        # Store positive overnight delta as target feature
        # Drop rows with missing target feature
        # Remove confounding features
        ###

        logger.info("Preparing data...")
        data["target"] = (data["overnight_delta"] > 0).astype(int)
        data = data.dropna(subset=["overnight_delta"])
        data = data.drop(columns=CONFOUND_FEATURES, errors="ignore")

        logger.info(f"✓ Loaded {len(data)} rows")

        return data

    def prepare_features(self, data):
        """
        Prepare features for training from dataset

        Parameters:
            1) data (type): fill here

        Returns:
            1) X (type): fill here
            2) y (type): fill here
        """

        # filter out non-safe features and fill missing/non-finite values
        X = prepare_model_features(data, self.feature_names)
        y = data["target"]

        return X, y

    def train(self, X_train, y_train):
        """
        Train model.

        Parameters:
            1) X_train (type): fill here
            2) y_train (type): fill here
        """

        logger.info("Training model...")

        # base model
        base_model = DecisionTreeClassifier(
            max_depth=self.config.get("max_depth", 5),
            min_samples_leaf=self.config.get("min_samples_leaf", 10),
            min_samples_split=self.config.get("min_samples_split", 20),
            class_weight=self.config.get("class_weight", "balanced"),
            random_state=self.config.get("random_state", 42),
        )

        # TimeSeriesSplit cross validation to prevent look-ahead leakage
        tscv = TimeSeriesSplit(n_splits=self.config.get("cv", 5))

        # Check for hyperparameter tuning (Task 5.4)
        tuning_config = self.config.get("tuning", {})
        if tuning_config.get("enabled", False):
            logger.info("Hyperparameter tuning enabled. Running GridSearchCV...")
            param_grid = tuning_config.get("param_grid", {
                "max_depth": [3, 5, 8, 12],
                "min_samples_leaf": [5, 10, 20],
                "min_samples_split": [10, 20, 40]
            })
            
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=tscv,
                scoring="accuracy",
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            logger.info(f"✓ Best parameters: {self.best_params}")
            logger.info(f"✓ Best CV accuracy: {grid_search.best_score_:.3f}")
        else:
            self.model = base_model
            cv_score = cross_val_score(self.model, X_train, y_train, cv=tscv)
            logger.info(f"✓ CV accuracy: {cv_score.mean():.3f} (+/- {cv_score.std():.3f})")
            self.model.fit(X_train, y_train)
            self.best_params = None

        self.trained_date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

        logger.info(f"✓ Model trained")
        logger.info(f"  - Tree depth: {self.model.get_depth()}")
        logger.info(f"  - Leaves: {self.model.get_n_leaves()}")

        return self

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Parameters:
            1) X_test (type): fill here
            2) y_test (type): fill here
        """

        logger.info("\nEvaluating model...")

        # predict labels for test set and store probabilities
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # calculate performance metrics and store as attribute
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        self.metrics = {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        }

        logger.info(f"✓ Test accuracy: {accuracy:.3f}")
        logger.info(f"\nConfusion Matrix:\n{cm}")

        # store feature importance
        importance = pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

        logger.info(f"\nTop 5 Important Features:")

        for ft, imp in list(importance.items())[:5]:
            logger.info(f"  {ft:30s}: {imp:.4f}")

        self.metrics["feature_importance"] = importance.to_dict()

        return self.metrics

    def save_model(self, save_path):
        """
        Save model bundle to input path.

        Parameters:
            1) save_path (str): Path to save bundle

        Returns:
            1) save_path (str): Path to save bundle
        """

        bundle = {
            "model": self.model,
            "features": self.feature_names,
            "threshold": self.config["probability_threshold"],
            "target": "overnight_delta_positive",
            "trained_date": self.trained_date,
            "config": self.config,
            "metrics": self.metrics,
        }

        # confirm path to save model bundle exists
        save_path = os.path.expanduser(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(bundle, save_path)

        logger.info(f"\n✓ Model saved to: {save_path}")

        return save_path


def train_model(data_path, save_path, config=None):
    """
    Train overnight delta prediction model.

    Parameters:
        1) data_path (str): Path to processed data for training
        2) save_path (str): Path to save model
        3) config (dict): Model configuration
    """

    logger.info("=" * 80)
    logger.info("OVERNIGHT DIRECTION MODEL TRAINING")
    logger.info("=" * 80)

    # initialize model
    model = OvernightDirectionModel(config=config)

    # load data
    data = model.load_data(data_path)

    # prepare features
    X, y = model.prepare_features(data)

    # split dataset into train/test sets without shuffling
    test_size = model.config["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    logger.info(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # train, evaluate, save model
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.save_model(save_path)

    logger.info("\n" + "=" * 80)
    logger.info("✓ TRAINING COMPLETE")
    logger.info("=" * 80)

    return model
