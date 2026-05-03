# qusa/qusa/model/evaluate.py

"""
Evaluate model performance.
"""

import joblib
import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from qusa.model.train import prepare_model_features


class ModelEvaluator:
    """
    Class to evaluate trained ML model.
    """

    def __init__(self, model_path):
        """
        Class constructor.

        Parameters:
            1) model_path (str): Path to saved/trained model
        """

        self.model_path = os.path.expanduser(model_path)
        self._load_model()

    def _load_model(self):
        """
        Load saved/trained model.
        """

        # load model bundle from path
        bundle = joblib.load(self.model_path)

        self.model = bundle["model"]
        self.features = bundle["features"]
        self.threshold = bundle["threshold"]

        print(f"✓ Model loaded")

        return

    def evaluate(self, test_data_path):
        """
        Evaluate model on test data.

        Parameters:
            1) test_data_path (str): Path to test data set

        Returns:
            1) metrics (dict): Evaluation metrics
        """

        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)

        # load test data
        data = pd.read_csv(os.path.expanduser(test_data_path))

        # define target feature if not present
        if ("target" not in data.columns) and ("overnight_delta" in data.columns):
            data["target"] = (data["overnight_delta"] > 0).astype(int)
        elif "target" not in data.columns:
            raise KeyError("Neither 'target' nor 'overnight_delta' found in test data.")

        # define target feature
        y_target = (data["target"] > 0).astype(int)

        # extract features and fill missing/non-finite values
        X = prepare_model_features(data, self.features)

        # predict
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]

        # calculate metrics
        metrics = self._calculate_metrics(y_target, y_pred, y_prob)
        metrics["calibration"] = self._analyze_calibration(y_target, y_prob)
        self._print_metrics(metrics)

        return metrics

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """
        Calculate performance metrics of trained model.

        Parameters:
            1) y_true (type): fill here
            2) y_pred (type): fill here
            3) y_prob (type): fill here
        """

        # define basic metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm
        metrics["true_negatives"] = int(cm[0, 0])
        metrics["false_positives"] = int(cm[0, 1])
        metrics["false_negatives"] = int(cm[1, 0])
        metrics["true_positives"] = int(cm[1, 1])

        # define filter for high confidence predictions
        mask_high_confidence = (y_prob >= self.threshold) | (
            y_prob <= (1 - self.threshold)
        )

        # handle high confidence predictions
        if mask_high_confidence.sum() > 0:
            metrics["high_confidence_coverage"] = mask_high_confidence.mean()
            metrics["high_confidence_accuracy"] = accuracy_score(
                y_true[mask_high_confidence], y_pred[mask_high_confidence]
            )
        # otherwise prediction has low confidence
        else:
            metrics["high_confidence_coverage"] = 0
            metrics["high_confidence_accuracy"] = 0

        return metrics

    @staticmethod
    def _analyze_calibration(y_true, y_prob):
        """
        Evaluate model prediction calibration.

        Parameters:
            1) y_true (type): fill here
            2) y_pred (type): fill here

        Return:
            1) calibration (type): fill here
        """

        # define probability boundaries
        bins = [0.0, 0.4, 0.5, 0.6, 0.7, 1.0]

        # merge actual, predicted labels by bins
        df = pd.DataFrame(
            {"y_true": y_true, "y_prob": y_prob, "bin": pd.cut(y_prob, bins=bins)}
        )

        calibration = df.groupby(["bin"]).agg(
            count=("y_true", "size"),
            actual_rate=("y_prob", "mean"),
            predicted_rate=("y_prob", "mean"),
        )

        return calibration

    def _print_metrics(self, metrics):
        """
        Pretty print model evaluation metrics.

        Parameters:
            1) metrics (dict): fill here
        """

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")

        if "confusion_matrix" in metrics:
            print(f"\nConfusion Matrix:")
            print(
                f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}"
            )
            print(
                f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}"
            )

        print(f"\nHigh-Confidence Predictions (>= {self.threshold}):")
        print(f"  Coverage: {metrics['high_confidence_coverage']:.1%}")
        print(f"  Accuracy: {metrics['high_confidence_accuracy']:.3f}")

        if "calibration" in metrics:
            print(f"\nProbability Calibration:")
            print(metrics["calibration"])

        return


def evaluate_model(model_path, eval_data_path):
    """
    Evaluate trained model on test data set.

    Parameters:
        1) model_path (str): Path to saved/trained model
        2) eval_data_path (str): Path to test dataset

    Returns:
        1) evaluation (dict): Evaluation metrics
    """

    # initialize class and run on model
    evaluator = ModelEvaluator(model_path)
    metrics = evaluator.evaluate(eval_data_path)

    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE")
    print("=" * 80)

    return metrics
