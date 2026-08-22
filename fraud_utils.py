import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def apply_threshold(probabilities, threshold):
    """Convert fraud probabilities into predicted classes."""

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    probabilities = pd.Series(probabilities)

    try:
        probabilities = pd.to_numeric(
            probabilities,
            errors="raise",
        )
    except (ValueError, TypeError):
        raise ValueError(
            "Probabilities must contain numeric values."
        )

    if probabilities.isna().any():
        raise ValueError(
            "Probabilities cannot contain missing values."
        )

    if (
        probabilities.lt(0).any()
        or probabilities.gt(1).any()
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

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
