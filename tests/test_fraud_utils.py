import pandas as pd

from fraud_utils import (
    apply_threshold,
    calculate_classification_metrics,
)


def test_apply_threshold():
    probabilities = [0.2, 0.5, 0.8]

    predictions = apply_threshold(
        probabilities,
        0.5,
    )

    expected = pd.Series([0, 1, 1])

    assert predictions.equals(expected)


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
