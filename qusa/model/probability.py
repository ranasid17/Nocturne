"""Class-label-aware probability extraction shared by model workflows."""

import numpy as np


def probability_of_up(model, features):
    """Return class-one probabilities, including zero for legacy one-class models."""
    probabilities = model.predict_proba(features)
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        return np.zeros(len(features))
    return probabilities[:, classes.index(1)]
