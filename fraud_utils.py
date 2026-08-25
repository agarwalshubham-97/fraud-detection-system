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
    actual = pd.Series(actual)

    if actual.isna().any():
        raise ValueError(
            "Actual values cannot contain missing values."
        )

    if not actual.isin([0, 1]).all():
        raise ValueError(
            "Actual values must contain only 0 and 1."
        )

    if len(actual) != len(probabilities):
        raise ValueError(
            "Actual values and probabilities must have the same length."
        )

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
def validate_transaction_data(
    data,
    feature_names,
):
    """Validate transaction data before model prediction."""

    missing_features = [
        feature
        for feature in feature_names
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    transaction_data = data[feature_names].copy()

    if transaction_data.isna().any().any():
        missing_columns = transaction_data.columns[
            transaction_data.isna().any()
        ].tolist()

        raise ValueError(
            "Missing values in: "
            + ", ".join(missing_columns)
        )

    try:
        transaction_data = transaction_data.apply(
            pd.to_numeric,
            errors="raise",
        )
    except (ValueError, TypeError):
        raise ValueError(
            "All required feature values must be numeric."
        )

    return transaction_data