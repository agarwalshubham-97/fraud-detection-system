import pandas as pd
import joblib
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


def validate_model_config(config):
    """Validate the model configuration."""

    if "threshold" not in config:
        raise ValueError(
            "Model configuration must contain 'threshold'."
        )

    threshold = config["threshold"]

    if not isinstance(threshold, (int, float)):
        raise ValueError(
            "Threshold must be numeric."
        )

    if not 0 < threshold < 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    return True


def load_model_artifacts(
    model_path,
    feature_names_path,
    config_path,
):
    """Load and validate model artifacts."""

    model = joblib.load(model_path)

    feature_names = joblib.load(
        feature_names_path
    )

    config = joblib.load(config_path)

    if not feature_names:
        raise ValueError(
            "Feature names cannot be empty."
        )

    validate_model_config(config)

    return model, feature_names, config

def load_evaluation_data(file_path):
    """Load and validate model evaluation data."""

    evaluation_data = pd.read_csv(file_path)

    required_columns = [
        "Actual",
        "Probability",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in evaluation_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Evaluation data is missing required columns: "
            + ", ".join(missing_columns)
        )

    if evaluation_data[required_columns].isnull().any().any():
        raise ValueError(
            "Evaluation data cannot contain missing values."
        )

    if not pd.api.types.is_numeric_dtype(
        evaluation_data["Actual"]
    ):
        raise ValueError(
            "Actual values must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(
        evaluation_data["Probability"]
    ):
        raise ValueError(
            "Probability values must be numeric."
        )

    if not evaluation_data["Actual"].isin([0, 1]).all():
        raise ValueError(
            "Actual values must be either 0 or 1."
        )

    if not evaluation_data["Probability"].between(
        0, 1
    ).all():
        raise ValueError(
            "Probability values must be between 0 and 1."
        )

    return evaluation_data