import pandas as pd
import pytest

from fraud_utils import (
    apply_threshold,
    calculate_classification_metrics,
    validate_transaction_data,
)
def test_apply_threshold_invalid_threshold():
    probabilities = [0.2, 0.5, 0.8]

    with pytest.raises(ValueError):
        apply_threshold(
            probabilities,
            -0.1,
        )

    with pytest.raises(ValueError):
        apply_threshold(
            probabilities,
            1.1,
        )

def test_apply_threshold_invalid_probabilities():
    with pytest.raises(ValueError):
        apply_threshold(
            [-0.1, 0.5, 0.8],
            0.5,
        )
    with pytest.raises(ValueError):
        apply_threshold(
            [0.2, 0.5, 1.1],
            0.5,
        )
def test_apply_threshold_non_numeric_probability():
    with pytest.raises(ValueError):
        apply_threshold(
            [0.2, "hello", 0.8],
            0.5,
        )
def test_apply_threshold_missing_probability():
    with pytest.raises(ValueError):
        apply_threshold(
            [0.2, None, 0.8],
            0.5,
        )

def test_apply_threshold():
    probabilities = [0.2, 0.5, 0.8]

    predictions = apply_threshold(
        probabilities,
        0.5,
    )

    expected = pd.Series([0, 1, 1])

    assert predictions.equals(expected)

def test_apply_threshold_boundary():
    probabilities = [0.49, 0.50, 0.51]

    predictions = apply_threshold(
        probabilities,
        0.50,
    )

    expected = pd.Series([0, 1, 1])

    assert predictions.equals(expected)

def test_classification_metrics_invalid_actual():
    with pytest.raises(ValueError):
        calculate_classification_metrics(
            [0, 1, 2],
            [0.1, 0.8, 0.9],
            0.5,
        )


def test_classification_metrics_missing_actual():
    with pytest.raises(ValueError):
        calculate_classification_metrics(
            [0, None, 1],
            [0.1, 0.5, 0.9],
            0.5,
        )

def test_classification_metrics_mismatched_lengths():
    with pytest.raises(ValueError):
        calculate_classification_metrics(
            [0, 1, 1],
            [0.2, 0.8],
            0.5,
        )

def test_classification_metrics():
    actual = [0, 0, 1, 1]
    probabilities = [0.1, 0.4, 0.6, 0.9]

    metrics = calculate_classification_metrics(
        actual,
        probabilities,
        0.5,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
def test_validate_transaction_data_missing_features():
    data = pd.DataFrame(
        {
            "Time": [100],
            "V1": [0.5],
        }
    )

    feature_names = [
        "Time",
        "V1",
        "Amount",
    ]

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        validate_transaction_data(
            data,
            feature_names,
        )


def test_validate_transaction_data_missing_values():
    data = pd.DataFrame(
        {
            "Time": [100, 200],
            "V1": [0.5, None],
            "Amount": [50.0, 75.0],
        }
    )

    feature_names = [
        "Time",
        "V1",
        "Amount",
    ]

    with pytest.raises(
        ValueError,
        match="Missing values",
    ):
        validate_transaction_data(
            data,
            feature_names,
        )


def test_validate_transaction_data_non_numeric():
    data = pd.DataFrame(
        {
            "Time": [100],
            "V1": ["invalid"],
            "Amount": [50.0],
        }
    )

    feature_names = [
        "Time",
        "V1",
        "Amount",
    ]

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        validate_transaction_data(
            data,
            feature_names,
        )


def test_validate_transaction_data_valid_data():
    data = pd.DataFrame(
        {
            "Time": [100, 200],
            "V1": [0.5, -1.2],
            "Amount": [50.0, 75.0],
            "Extra_Column": ["A", "B"],
        }
    )

    feature_names = [
        "Time",
        "V1",
        "Amount",
    ]

    result = validate_transaction_data(
        data,
        feature_names,
    )

    assert list(result.columns) == feature_names
    assert len(result) == 2
    assert "Extra_Column" not in result.columns

