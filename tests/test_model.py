import joblib
import numpy as np
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


def test_model_loads():
    """Check that the trained model loads successfully."""
    assert model is not None


def test_feature_count():
    """Check that the model uses 30 features."""
    assert len(feature_names) == 30


def test_prediction_output():
    """Check that predictions return valid classes."""

    sample_data = np.zeros(
        (1, len(feature_names))
    )

    prediction = model.predict(sample_data)

    assert prediction[0] in [0, 1]


def test_probability_range():
    """Check that fraud probability is between 0 and 1."""

    sample_data = np.zeros(
        (1, len(feature_names))
    )

    probability = model.predict_proba(
        sample_data
    )[0][1]

    assert 0 <= probability <= 1


def test_threshold_valid():
    """Check that the configured threshold is valid."""

    threshold = config["threshold"]

    assert 0 < threshold < 1
