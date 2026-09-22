# qusa/qusa/model/__init__.py

"""
Machine learning model for overnight price prediction
"""

from .backtest import ModelBacktester
from .evaluate import evaluate_model
from .predict import make_prediction, LivePredictor
from .train import train_model, OvernightDirectionModel

__all__ = [
    "ModelBacktester",
    "evaluate_model",
    "LivePredictor",
    "train_model",
    "make_prediction",
    "OvernightDirectionModel",
]


def __getattr__(name):
    """Keep legacy report imports lazy so prediction startup needs no LLM client."""
    if name.startswith("generate_") and name.endswith("_report"):
        from . import reports
        return getattr(reports, name)
    raise AttributeError(name)
