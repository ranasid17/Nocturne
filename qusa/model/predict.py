# qusa/qusa/model/predict.py

"""
Use trained model to make live
predictions on most-recent trading
day data.
"""

import logging
import joblib
import math
import os
import numpy as np
import pandas as pd

from datetime import datetime
from qusa.model.dataset import DatasetContractError, prepare_model_features
from qusa.utils.formatting import format_prediction_card

logger = logging.getLogger(__name__)


class LivePredictor:
    """
    Make predictions on live market data.
    """

    def __init__(self, model_path):
        """
        Class constructor.

        Parameters;
            1) model_path (str): Path to saved model
        """
        self.model_path = os.path.expanduser(model_path)
        self._load_model()

    def _load_model(self):
        """
        Load trained model.
        """

        bundle = joblib.load(self.model_path)

        self.model = bundle["model"]
        self.features = bundle["features"]
        self.threshold = bundle["threshold"]
        self.trained_date = bundle.get("trained_date", "Unknown")
        self.feature_manifest_version = bundle.get("feature_manifest_version", "legacy")
        self.model_id = bundle.get("model_id")

        if not isinstance(self.features, list) or not self.features:
            raise ValueError("Model bundle has no usable feature manifest.")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Model bundle feature manifest contains duplicates.")
        model_feature_names = getattr(self.model, "feature_names_in_", None)
        if model_feature_names is not None and list(model_feature_names) != self.features:
            raise ValueError("Model feature schema does not match its saved manifest.")
        model_feature_count = getattr(self.model, "n_features_in_", None)
        if model_feature_count is not None and model_feature_count != len(self.features):
            raise ValueError("Model feature count does not match its saved manifest.")

        logger.info(f"✓ Model loaded (trained: {self.trained_date})")

        return

    def predict(self, data, volatility_filter=None):
        """
        Make prediction on most-recent data

        Parameters:
            1) data (pd.DataFrame): Data with features
            2) volatility_filter (dict, optional): Volatility filter settings

        Returns:
            1) prediction (dict): Results
        """

        if data.empty:
            raise ValueError("Prediction data has no rows.")

        # get latest row
        latest = data.tail(1)

        try:
            X = prepare_model_features(latest, self.features, strict=True)
        except DatasetContractError as exc:
            raise ValueError(str(exc)) from exc

        # predict labels and probabilities
        y_pred = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        classes = list(getattr(self.model, "classes_", []))
        if 1 not in classes:
            y_prob = 0.0
        else:
            y_prob = float(probabilities[classes.index(1)])

        # interpret prediction
        if y_pred == 1:
            direction = "UP ⬆"
        else:
            direction = "DOWN ⬇"

        # interpret prediction probability
        if (y_prob >= self.threshold) or (y_prob <= (1 - self.threshold)):
            confidence = "HIGH"
        else:
            confidence = "LOW"

        # check volatility filter
        vol_triggered = False
        atr_pct = None
        volatility_state = "disabled"
        volatility_threshold = None
        if volatility_filter and volatility_filter.get("enabled", False):
            volatility_threshold = volatility_filter.get("max_atr_pct", 100.0)
            if (
                isinstance(volatility_threshold, bool)
                or not isinstance(volatility_threshold, (int, float))
                or not math.isfinite(float(volatility_threshold))
                or float(volatility_threshold) < 0
            ):
                raise ValueError("Volatility threshold must be a finite non-negative number.")
            volatility_threshold = float(volatility_threshold)
            raw_atr = latest["atr_pct"].iloc[0] if "atr_pct" in latest.columns else None
            if (
                raw_atr is None
                or isinstance(raw_atr, bool)
                or not isinstance(raw_atr, (int, float, np.number))
                or not np.isfinite(raw_atr)
                or raw_atr < 0
            ):
                volatility_state = "unavailable"
                logger.warning("Volatility filter is unavailable because atr_pct is missing or invalid.")
            else:
                atr_pct = float(raw_atr)
                vol_triggered = atr_pct > volatility_threshold
                volatility_state = "blocked" if vol_triggered else "pass"

        # store prediction metadata in dictionary
        prediction = {
            "date": latest["date"].iloc[0] if "date" in latest.columns else None,
            "prediction": y_pred,
            "direction": direction,
            "probability_up": y_prob,
            "confidence": confidence,
            "volatility_filter_triggered": vol_triggered,
            "atr_pct": atr_pct,
            "volatility_state": volatility_state,
            "volatility_threshold": volatility_threshold,
            "model_id": self.model_id,
            "feature_manifest_version": self.feature_manifest_version,
        }

        return prediction

    @staticmethod
    def print_prediction(prediction, ticker=None, logger_obj=None):
        """
        Print formatted prediction.

        Parameters:
            1) prediction (dict): Model prediction on most-recent data
            2) ticker (str): Ticker symbol
            3) logger_obj (logging.Logger): Logger instance to use
        """
        log = logger_obj or logger
        card = format_prediction_card(prediction, ticker=ticker)
        
        for line in card.split("\n"):
            log.info(line)

        return


def make_prediction(model_path, data_path, ticker=None, logger_obj=None, volatility_filter=None):
    """
    Use model to predict on most-recent
    input data.

    Parameters:
        1) model_path (str): Path to saved/trained model
        2) data_path (str): Path to data with required features
        3) ticker (str): Ticker symbol
        4) logger_obj (logging.Logger): Logger instance to use
        5) volatility_filter (dict, optional): Volatility filter settings

    Returns:
        1) prediction (dict): Model prediction
    """

    # load predictor
    predictor = LivePredictor(model_path)

    # load data and confirm datetime stamp
    data = pd.read_csv(os.path.expanduser(data_path))
    data["date"] = pd.to_datetime(data["date"])

    # make prediction
    prediction = predictor.predict(data, volatility_filter=volatility_filter)

    # print prediction
    predictor.print_prediction(prediction, ticker=ticker, logger_obj=logger_obj)

    return prediction
