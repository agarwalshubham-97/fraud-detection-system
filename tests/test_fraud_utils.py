import pandas as pd
import pytest
import joblib

class PredictOnlyModel:
    def predict(self, data):
        return [0]

class TestModel:
    def predict(self, data):
        return [0]

    def predict_proba(self, data):
        return [[1.0, 0.0]]

from fraud_utils import (
    apply_threshold,
    calculate_classification_metrics,
    validate_transaction_data,
    validate_model_config,
    load_model_artifacts,
    load_evaluation_data,
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
def test_validate_model_config_missing_threshold():
    """Check that a missing threshold raises an error."""

    config = {}

    with pytest.raises(
        ValueError,
        match="Model configuration must contain 'threshold'.",
    ):
        validate_model_config(config)


def test_validate_model_config_non_numeric_threshold():
    """Check that a non-numeric threshold raises an error."""

    config = {"threshold": "invalid"}

    with pytest.raises(
        ValueError,
        match="Threshold must be numeric.",
    ):
        validate_model_config(config)


def test_validate_model_config_below_range():
    """Check that a threshold below the valid range raises an error."""

    config = {"threshold": -0.1}

    with pytest.raises(
        ValueError,
        match="Threshold must be between 0 and 1.",
    ):
        validate_model_config(config)


def test_validate_model_config_above_range():
    """Check that a threshold above the valid range raises an error."""

    config = {"threshold": 1.1}

    with pytest.raises(
        ValueError,
        match="Threshold must be between 0 and 1.",
    ):
        validate_model_config(config)


def test_validate_model_config_valid():
    """Check that a valid model configuration passes validation."""

    config = {"threshold": 0.5}

    assert validate_model_config(config) is True


def test_load_model_artifacts_valid(tmp_path):
    """Check that valid model artifacts load successfully."""

    model_path = tmp_path / "model.pkl"
    feature_names_path = tmp_path / "features.pkl"
    config_path = tmp_path / "config.pkl"

    test_model = TestModel()
    test_feature_names = ["Time", "V1", "Amount"]
    test_config = {"threshold": 0.5}

    joblib.dump(test_model, model_path)
    joblib.dump(
        test_feature_names,
        feature_names_path,
    )
    joblib.dump(
        test_config,
        config_path,
    )

    model, feature_names, config = (
        load_model_artifacts(
            model_path,
            feature_names_path,
            config_path,
        )
    )

    assert isinstance(model, TestModel)
    assert feature_names == test_feature_names
    assert config == test_config


def test_load_model_artifacts_empty_feature_names(
    tmp_path,
):
    """Check that empty feature names raise an error."""

    model_path = tmp_path / "model.pkl"
    feature_names_path = tmp_path / "features.pkl"
    config_path = tmp_path / "config.pkl"

    joblib.dump({"name": "test_model"}, model_path)
    joblib.dump([], feature_names_path)
    joblib.dump(
        {"threshold": 0.5},
        config_path,
    )

    with pytest.raises(
        ValueError,
        match="Feature names cannot be empty",
    ):
        load_model_artifacts(
            model_path,
            feature_names_path,
            config_path,
        )


def test_load_model_artifacts_invalid_config(
    tmp_path,
):
    """Check that an invalid configuration raises an error."""

    model_path = tmp_path / "model.pkl"
    feature_names_path = tmp_path / "features.pkl"
    config_path = tmp_path / "config.pkl"

    joblib.dump({"name": "test_model"}, model_path)
    joblib.dump(
        ["Time", "V1", "Amount"],
        feature_names_path,
    )
    joblib.dump({}, config_path)

    with pytest.raises(
        ValueError,
        match="Model configuration must contain 'threshold'",
    ):
        load_model_artifacts(
            model_path,
            feature_names_path,
            config_path,
        )

def test_load_evaluation_data_valid(tmp_path):
    """Check that valid evaluation data loads successfully."""

    file_path = tmp_path / "evaluation.csv"

    data = pd.DataFrame(
        {
            "Actual": [0, 1, 0, 1],
            "Probability": [0.10, 0.90, 0.20, 0.80],
        }
    )

    data.to_csv(file_path, index=False)

    result = load_evaluation_data(file_path)

    assert list(result.columns) == [
        "Actual",
        "Probability",
    ]
    assert len(result) == 4


def test_load_evaluation_data_missing_actual(tmp_path):
    """Check that a missing Actual column raises an error."""

    file_path = tmp_path / "evaluation.csv"

    data = pd.DataFrame(
        {
            "Probability": [0.10, 0.90],
        }
    )

    data.to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="Actual",
    ):
        load_evaluation_data(file_path)


def test_load_evaluation_data_missing_probability(tmp_path):
    """Check that a missing Probability column raises an error."""

    file_path = tmp_path / "evaluation.csv"

    data = pd.DataFrame(
        {
            "Actual": [0, 1],
        }
    )

    data.to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="Probability",
    ):
        load_evaluation_data(file_path)


def test_load_evaluation_data_missing_values(tmp_path):
    """Check that missing evaluation values raise an error."""

    file_path = tmp_path / "evaluation.csv"

    data = pd.DataFrame(
        {
            "Actual": [0, 1],
            "Probability": [0.10, None],
        }
    )

    data.to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        load_evaluation_data(file_path)

def test_load_evaluation_data_non_numeric_actual(tmp_path):
    """Check that non-numeric Actual values raise an error."""

    file_path = tmp_path / "evaluation.csv"

    data = pd.DataFrame(
        {
            "Actual": ["normal", "fraud"],
            "Probability": [0.10, 0.90],
        }
    )

    data.to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="Actual values must be numeric",
    ):
        load_evaluation_data(file_path)


def test_load_evaluation_data_non_numeric_probability(tmp_path):
    """Check that non-numeric Probability values raise an error."""

    file_path = tmp_path / "evaluation.csv"

    data = pd.DataFrame(
        {
            "Actual": [0, 1],
            "Probability": ["low", "high"],
        }
    )

    data.to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="Probability values must be numeric",
    ):
        load_evaluation_data(file_path)

def test_load_evaluation_data_invalid_actual(tmp_path):
    """Check that Actual values must be 0 or 1."""

    file_path = tmp_path / "evaluation.csv"

    data = pd.DataFrame(
        {
            "Actual": [0, 1, 2],
            "Probability": [0.10, 0.90, 0.50],
        }
    )

    data.to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="Actual values must be either 0 or 1",
    ):
        load_evaluation_data(file_path)


def test_load_evaluation_data_invalid_probability(tmp_path):
    """Check that Probability values must be between 0 and 1."""

    file_path = tmp_path / "evaluation.csv"

    data = pd.DataFrame(
        {
            "Actual": [0, 1, 0],
            "Probability": [0.10, 1.20, 0.50],
        }
    )

    data.to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="Probability values must be between 0 and 1",
    ):
        load_evaluation_data(file_path)

def test_load_model_artifacts_missing_file(tmp_path):
    """Check that missing model artifacts raise FileNotFoundError."""

    model_path = tmp_path / "model.pkl"
    feature_names_path = tmp_path / "features.pkl"
    config_path = tmp_path / "config.pkl"

    joblib.dump(
        ["Time", "V1", "Amount"],
        feature_names_path,
    )

    joblib.dump(
        {"threshold": 0.5},
        config_path,
    )

    with pytest.raises(FileNotFoundError):
        load_model_artifacts(
            model_path,
            feature_names_path,
            config_path,
        )

def test_load_model_artifacts_empty_model(tmp_path):
    """Check that an empty model artifact raises an error."""

    model_path = tmp_path / "model.pkl"
    feature_names_path = tmp_path / "features.pkl"
    config_path = tmp_path / "config.pkl"

    joblib.dump(None, model_path)

    joblib.dump(
        ["Time", "V1", "Amount"],
        feature_names_path,
    )

    joblib.dump(
        {"threshold": 0.5},
        config_path,
    )

    with pytest.raises(
        ValueError,
        match="Model cannot be empty",
    ):
        load_model_artifacts(
            model_path,
            feature_names_path,
            config_path,
        )

def test_load_model_artifacts_invalid_model(tmp_path):
    """Check that an invalid model artifact raises an error."""

    model_path = tmp_path / "model.pkl"
    feature_names_path = tmp_path / "features.pkl"
    config_path = tmp_path / "config.pkl"

    joblib.dump(
        {"name": "not_a_model"},
        model_path,
    )

    joblib.dump(
        ["Time", "V1", "Amount"],
        feature_names_path,
    )

    joblib.dump(
        {"threshold": 0.5},
        config_path,
    )

    with pytest.raises(
        ValueError,
        match="Model must provide a 'predict' method",
    ):
        load_model_artifacts(
            model_path,
            feature_names_path,
            config_path,
        )

def test_load_model_artifacts_missing_predict_proba(tmp_path):
    """Check that a model without predict_proba raises an error."""

    model_path = tmp_path / "model.pkl"
    feature_names_path = tmp_path / "features.pkl"
    config_path = tmp_path / "config.pkl"

    joblib.dump(
        PredictOnlyModel(),
        model_path,
    )

    joblib.dump(
        ["Time", "V1", "Amount"],
        feature_names_path,
    )

    joblib.dump(
        {"threshold": 0.5},
        config_path,
    )

    with pytest.raises(
        ValueError,
        match="Model must provide a 'predict_proba' method",
    ):
        load_model_artifacts(
            model_path,
            feature_names_path,
            config_path,
        )
