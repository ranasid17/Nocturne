# qusa/qusa/model/predict.py

"""
Use trained model to make live
predictions on most-recent trading
day data.
"""

import logging
import joblib
import os
import pandas as pd

from datetime import datetime
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

        logger.info(f"✓ Model loaded (trained: {self.trained_date})")

        return

    def predict(self, data):
        """
        Make prediction on most-recent data

        Parameters:
            1) data (pd.DataFrame): Data with features

        Returns:
            1) prediction (dict): Results
        """

        # get latest row
        latest = data.tail(1)

        # extract features
        X = latest[self.features].fillna(0)

        # predict labels and probabilities
        y_pred = self.model.predict(X)[0]
        y_prob = self.model.predict_proba(X)[0, 1]

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

        # store prediction metadata in dictionary
        prediction = {
            "date": latest["date"].iloc[0] if "date" in latest.columns else None,
            "prediction": y_pred,
            "direction": direction,
            "probability_up": y_prob,
            "confidence": confidence,
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


def make_prediction(model_path, data_path, ticker=None, logger_obj=None):
    """
    Use model to predict on most-recent
    input data.

    Parameters:
        1) model_path (str): Path to saved/trained model
        2) data_path (str): Path to data with required features
        3) ticker (str): Ticker symbol
        4) logger_obj (logging.Logger): Logger instance to use

    Returns:
        1) prediction (dict): Model prediction
    """

    # load predictor
    predictor = LivePredictor(model_path)

    # load data and confirm datetime stamp
    data = pd.read_csv(os.path.expanduser(data_path))
    data["date"] = pd.to_datetime(data["date"])

    # make prediction
    prediction = predictor.predict(data)

    # print prediction
    predictor.print_prediction(prediction, ticker=ticker, logger_obj=logger_obj)

    return prediction
