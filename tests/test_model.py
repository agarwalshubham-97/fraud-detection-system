import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Load project files
model = joblib.load(
    BASE_DIR / "models" / "fraud_detection_model.pkl"
)

feature_names = joblib.load(
    BASE_DIR / "models" / "feature_names.pkl"
)

config = joblib.load(
    BASE_DIR / "models" / "model_config.pkl"
)


def create_sample_data():
    """Create a sample transaction using all model features."""

    return pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )


def test_model_loads():
    """Check that the trained model loads successfully."""

    assert model is not None


def test_model_prediction_methods():
    """Check that the model provides required prediction methods."""

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_feature_count():
    """Check that the model uses 30 features."""

    assert len(feature_names) == 30


def test_feature_names_are_valid():
    """Check that feature names are non-empty strings."""

    assert feature_names

    assert all(
        isinstance(feature, str)
        for feature in feature_names
    )


def test_prediction_output():
    """Check that predictions return valid classes."""

    sample_data = create_sample_data()

    prediction = model.predict(sample_data)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_probability_output_shape():
    """Check that predict_proba returns the expected shape."""

    sample_data = create_sample_data()

    probabilities = model.predict_proba(sample_data)

    assert probabilities.shape == (1, 2)


def test_probability_range():
    """Check that fraud probability is between 0 and 1."""

    sample_data = create_sample_data()

    probability = model.predict_proba(
        sample_data
    )[0][1]

    assert 0 <= probability <= 1


def test_probability_sum():
    """Check that class probabilities sum to 1."""

    sample_data = create_sample_data()

    probabilities = model.predict_proba(
        sample_data
    )[0]

    assert np.isclose(
        probabilities.sum(),
        1.0
    )


def test_threshold_valid():
    """Check that the configured threshold is valid."""

    assert config is not None
    assert "threshold" in config

    threshold = config["threshold"]

    assert isinstance(
        threshold,
        (int, float),
    )

    assert 0 < threshold < 1

def test_batch_prediction_output():
    """Check that the model predicts multiple transactions correctly."""

    sample_data = pd.DataFrame(
        np.zeros((5, len(feature_names))),
        columns=feature_names
    )

    predictions = model.predict(sample_data)
    probabilities = model.predict_proba(sample_data)

    assert len(predictions) == 5
    assert probabilities.shape == (5, 2)

    assert all(
        prediction in [0, 1]
        for prediction in predictions
    )

    assert np.all(
        (probabilities >= 0)
        & (probabilities <= 1)
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0
    )

def test_model_prediction_with_different_transactions():
    """Check predictions for transactions with different values."""

    sample_data = pd.DataFrame(
        np.zeros((3, len(feature_names))),
        columns=feature_names
    )

    sample_data["Time"] = [100.0, 5000.0, 20000.0]
    sample_data["Amount"] = [10.0, 500.0, 5000.0]

    predictions = model.predict(sample_data)
    probabilities = model.predict_proba(sample_data)

    assert len(predictions) == 3
    assert probabilities.shape == (3, 2)

    assert all(
        prediction in [0, 1]
        for prediction in predictions
    )

    assert np.all(
        (probabilities >= 0)
        & (probabilities <= 1)
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0
    )
