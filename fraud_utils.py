import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def apply_threshold(probabilities, threshold):
    """Convert fraud probabilities into predicted classes."""
    probabilities = pd.Series(probabilities)

    return (probabilities >= threshold).astype(int)


def calculate_classification_metrics(
    actual,
    probabilities,
    threshold,
):
    """Calculate classification metrics using a threshold."""

    predictions = apply_threshold(
        probabilities,
        threshold,
    )

    return {
        "accuracy": accuracy_score(
            actual,
            predictions,
        ),
        "precision": precision_score(
            actual,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            actual,
            predictions,
            zero_division=0,
        ),
    }
