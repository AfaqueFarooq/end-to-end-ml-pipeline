import pytest
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.components.data_transformation import DataTransformation
from src.utils import save_object, evaluate_models

# Fixtures

@pytest.fixture
def sample_input():
    """A minimal valid input dataframe matching the expected feature schema."""
    return pd.DataFrame({
        'gender': ['male', 'female', 'male'],
        'race_ethnicity': ['group A', 'group B', 'group C'],
        'parental_level_of_education': ["bachelor's degree", 'some college', 'high school'],
        'lunch': ['standard', 'free/reduced', 'standard'],
        'test_preparation_course': ['none', 'completed', 'none'],
        'reading_score': [70, 85, 60],
        'writing_score': [65, 80, 55]
    })


@pytest.fixture
def sample_train_test():
    """Simple numeric train/test arrays for model evaluation tests."""
    np.random.seed(42)
    X_train = np.random.rand(100, 3)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.rand(100) * 0.1
    X_test = np.random.rand(20, 3)
    y_test = X_test[:, 0] * 2 + X_test[:, 1] * 3 + np.random.rand(20) * 0.1
    return X_train, y_train, X_test, y_test



# Data Transformation Tests


class TestDataTransformation:

    def test_preprocessor_returns_object(self):
        """Check that get_data_transformer_object returns a ColumnTransformer."""
        dt = DataTransformation()
        preprocessor = dt.get_data_transformer_object()
        assert preprocessor is not None

    def test_preprocessor_output_shape(self, sample_input):
        """Check that the preprocessor transforms input without errors and returns correct row count."""
        dt = DataTransformation()
        preprocessor = dt.get_data_transformer_object()
        result = preprocessor.fit_transform(sample_input)
        assert result.shape[0] == 3  # 3 input rows

    def test_preprocessor_handles_missing_values(self):
        """Check that the preprocessor handles NaN values gracefully via SimpleImputer."""
        dt = DataTransformation()
        preprocessor = dt.get_data_transformer_object()
        data_with_nan = pd.DataFrame({
            'gender': ['male', 'male', 'female'],
            'race_ethnicity': ['group A', 'group B', 'group A'],
            'parental_level_of_education': ["bachelor's degree", "bachelor's degree", 'high school'],
            'lunch': ['standard', 'free/reduced', 'standard'],
            'test_preparation_course': ['none', 'none', 'completed'],
            'reading_score': [70, np.nan, 60],
            'writing_score': [65, 80, np.nan]
        })
        result = preprocessor.fit_transform(data_with_nan)
        assert result.shape[0] == 3



# Utils Tests

class TestEvaluateModels:

    def test_evaluate_models_returns_dict(self, sample_train_test):
        """Check that evaluate_models returns a dictionary with model names as keys."""
        X_train, y_train, X_test, y_test = sample_train_test
        models = {"Linear Regression": LinearRegression()}
        params = {"Linear Regression": {}}
        report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
        assert isinstance(report, dict)
        assert "Linear Regression" in report

    def test_evaluate_models_r2_between_0_and_1(self, sample_train_test):
        """Check that R² scores returned are valid (between 0 and 1 for a sensible model)."""
        X_train, y_train, X_test, y_test = sample_train_test
        models = {"Linear Regression": LinearRegression()}
        params = {"Linear Regression": {}}
        report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
        score = report["Linear Regression"]
        assert 0 <= score <= 1

    def test_evaluate_models_multiple_models(self, sample_train_test):
        """Check that evaluate_models handles multiple models correctly."""
        X_train, y_train, X_test, y_test = sample_train_test
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor()
        }
        params = {
            "Linear Regression": {},
            "Decision Tree": {"max_depth": [3, 5]}
        }
        report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
        assert len(report) == 2



# Save Object Test

class TestSaveObject:

    def test_save_object_creates_file(self, tmp_path):
        """Check that save_object creates a file at the given path."""
        file_path = str(tmp_path / "test_model.pkl")
        model = LinearRegression()
        save_object(file_path, model)
        assert os.path.exists(file_path)