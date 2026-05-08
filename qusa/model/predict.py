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


class LivePredictor:
    """
    Make predictions on live market data.
    """

    def __init__(self, model_path, logger=None):
        """
        Class constructor.

        Parameters;
            1) model_path (str): Path to saved model
            2) logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
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

        self.logger.info(f"✓ Model loaded (trained: {self.trained_date})")

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
    def print_prediction(prediction):
        """
        Print formatted prediction.

        Parameters:
            1) prediction (dict): Model prediction on most-recent data
        """

        print("\n" + "=" * 80)
        print("PREDICTION")
        print("=" * 80)

        if prediction["date"]:
            print(f"Date: {prediction['date']}")

        print(f"Direction: {prediction['direction']}")
        print(f"Probability (UP): {prediction['probability_up']:.1%}")
        print(f"Confidence: {prediction['confidence']}")

        # handle cases with high prediction confidence
        if prediction["confidence"] == "HIGH":
            # handle case where high confidence positive prediction
            if prediction["prediction"] == 1:
                print("\n✓ STRONG BUY signal")
            # otherwise high confidence negative prediction
            else:
                print("\n✓ STRONG SELL signal")

        # otherwise low prediction confidence
        else:
            print("\n⚠ LOW CONFIDENCE - No clear signal")

        return


def make_prediction(model_path, data_path):
    """
    Use model to predict on most-recent
    input data.

    Parameters:
        1) model_path (str): Path to saved/trained model
        2) data_path (str): Path to data with required features

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
    predictor.print_prediction(prediction)

    return prediction
