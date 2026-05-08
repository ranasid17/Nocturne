# qusa/qusa/model/interpreter.py

"""
Interpret and explain trained models.
"""

import logging
import joblib
import numpy as np
import os
import pandas as pd

from pathlib import Path


class ModelInterpreter:
    """
    Extract insights and explanations from trained models.
    """

    def __init__(self, model_path, config=None, logger=None):
        """
        Initialize interpreter with model and optional config.

        Parameters:
            1) model_path (str): Path to saved model bundle
            2) config (dict, optional): Configuration dictionary
            3) logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        if not config:
            raise ValueError("config is required to initialize ModelInterpreter")

        self.model_path = Path(model_path).expanduser().resolve()
        self._load_model()

        interp_config = config.get("interpretation", {})
        self.top_n_features = interp_config.get("top_n_features", 10)
        self.max_decision_depth = interp_config.get("max_decision_depth", 10)
        self.low_importance_threshold = interp_config.get(
            "low_importance_threshold", 0.01
        )
        self.high_correlation_threshold = interp_config.get(
            "high_correlation_threshold", 0.9
        )
        self.low_confidence_threshold = interp_config.get(
            "low_confidence_threshold", 0.6
        )

    def _load_model(self):
        """
        Load the trained model from the specified path.
        """

        # load the model bundle
        bundle = joblib.load(self.model_path)

        # extract components and store as attributes
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.threshold = bundle.get("threshold", 0.6)
        self.trained_date = bundle.get("trained_date", "Unknown")
        self.metrics = bundle.get("metrics", {})

        self.logger.info(f"✓ Interpreter loaded model (trained: {self.trained_date})")

        return

    def analyze_feature_importance(self, data=None):
        """
        Analyze and return the top N features by importance.
        """

        # calculate feature importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # select the top N features
        top_indices = indices[: self.top_n_features]
        top_features = [self.features[i] for i in top_indices]
        top_importances = [importances[i] for i in top_indices]

        # create a DataFrame for visualization or further analysis
        feature_importance_df = pd.DataFrame(
            {"feature": top_features, "importance": top_importances}
        )

        return feature_importance_df

    def analyze_prediction_patterns(self, data):
        """
        Analyze patterns in model predictions.
        """

        # load and prepare data for prediction
        from qusa.model.train import prepare_model_features

        X = prepare_model_features(data, self.features)

        # generate predictions and probabilities
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]

        # merge predictions with original data
        analysis_df = data.copy()
        analysis_df["predicted_direction"] = y_pred
        analysis_df["prediction_probability"] = y_prob

        return analysis_df

    def identify_model_limitations(self, data=None, evaluation_metrics=None):
        """
        Identify potential limitations or issues with the model.
        """

        limitations = {
            "overfitting_risk": [],
            "data_quality_concerns": [],
            "reliability_issues": [],
        }

        # check for shallow or very deep trees
        if hasattr(self.model, "get_depth"):
            depth = self.model.get_depth()
            leaves = self.model.get_n_leaves()

            if depth < 2:
                limitations["data_quality_concerns"].append(
                    "Shallow tree suggests limited signal in features"
                )

            elif depth > 15:
                limitations["data_quality_concerns"].append(
                    "Deep tree suggests potential overfitting"
                )

            if leaves < 5:
                limitations["data_quality_concerns"].append(
                    "Few leaf nodes suggest model may be too simple"
                )

        # evaluate metrics if provided
        if evaluation_metrics:
            accuracy = evaluation_metrics.get("accuracy", 0)
            f1 = evaluation_metrics.get("f1", 0)

            if accuracy < 0.52:
                limitations["reliability_issues"].append(
                    f"Low accuracy ({accuracy:.1%}) suggests model is barely better than random"
                )

            if f1 < 0.4:
                limitations["reliability_issues"].append(
                    f"Low F1 score ({f1:.2f}) suggests poor balance between precision and recall"
                )

        return limitations

    def generate_interpretation_summary(self, data=None, evaluation_metrics=None):
        """
        Generate a summary dictionary of model interpretation.
        """

        summary = {
            "model_path": str(self.model_path),
            "trained_date": self.trained_date,
            "feature_count": len(self.features),
            "top_features": self.analyze_feature_importance().to_dict(orient="records"),
            "limitations": self.identify_model_limitations(
                data=data, evaluation_metrics=evaluation_metrics
            ),
        }

        return summary

    def print_interpretation(self, data=None, evaluation_metrics=None):
        """
        Print formatted model interpretation.
        """

        summary = self.generate_interpretation_summary(
            data=data, evaluation_metrics=evaluation_metrics
        )

        print("\n" + "=" * 80)
        print("MODEL INTERPRETATION SUMMARY")
        print("=" * 80)
        print(f"Model Path:   {summary['model_path']}")
        print(f"Trained Date: {summary['trained_date']}")
        print(f"Features:     {summary['feature_count']}")

        print("\n" + "-" * 80)
        print("TOP FEATURES BY IMPORTANCE")
        print("-" * 80)
        for i, feat in enumerate(summary["top_features"]):
            print(f"{i+1:2d}. {feat['feature']:30s} | {feat['importance']:.4f}")

        print("\n" + "-" * 80)
        print("MODEL LIMITATIONS & RISKS")
        print("-" * 80)

        has_issues = False
        for category, issues in summary["limitations"].items():
            if issues:
                has_issues = True
                print(f"\n{category.replace('_', ' ').title()}:")
                for issue in issues:
                    print(f"  ⚠ {issue}")

        if not has_issues:
            print("  ✓ No major limitations identified")

        print("\n" + "=" * 80)

        return
