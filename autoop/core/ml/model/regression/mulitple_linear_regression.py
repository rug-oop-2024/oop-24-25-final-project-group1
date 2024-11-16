import numpy as np
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
from pydantic import PrivateAttr
from typing import Dict
import numpy as np


class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression model implementation that inherits from the
    base "Model" class.

    Calculates the linear correlation between features (the observations) and
    a target variable (ground truth).

    Attributes:
    - _weights (np.ndarray): Stores the coefficients (weights) and intercept
      of the model.
    """
    _weights: np.ndarray = PrivateAttr(default=None)
    _artifact: Artifact = PrivateAttr(default=None)

    def __init__(self, name: str = "test_model", asset_path: str = "./tmp", version: str = "0.1", regularization: float = 0.0, **data):
        """
        Initializes the Multiple Linear Regression model with test-friendly defaults.

        Args:
            name (str): Model name, default "test_model" for testing.
            asset_path (str): Model asset path, default "./tmp".
            version (str): Model version, default "0.1".
            regularization (float): Regularization strength (default: 0.0).
        """
        super().__init__(name=name, asset_path=asset_path, version=version, **data)
        self._parameters = {
            "regularization": regularization
        }
        self._type = "regression"

    @property
    def weights(self) -> Dict:
        """
        Returns a copy of the model weights.
        """
        return self._weights.copy() if self._weights is not None else None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the multiple linear regression model to the provided training data.
        Applies L2 regularization if the regularization parameter is greater than 0.

        Args:
            observations (np.ndarray): Feature matrix of shape (n_samples,
            p_features).
            ground_truth (np.ndarray): Target vector of shape (n_samples,).
        """
        regularization = self._parameters.get('regularization', 0.0)
        observations_b: np.ndarray = np.c_[
            observations, np.ones(observations.shape[0])
        ]

        XtX = np.matmul(observations_b.T, observations_b)
        XtY = np.matmul(observations_b.T, ground_truth)

        if regularization > 0:
            identity_matrix = np.eye(XtX.shape[0])
            identity_matrix[-1, -1] = 0
            XtX += regularization * identity_matrix

        try:
            XtX_inv = np.linalg.pinv(XtX)
            self._weights = np.matmul(XtX_inv, XtY)
            self._parameters['weights'] = self._weights
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error inverting matrix during training: {e}")
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the values for the provided observations.

        Args:
            observations (np.ndarray): Feature matrix of shape (n_samples,
            p_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        if self._weights is None:
            raise ValueError("Model has not been trained yet. Please fit the model before predicting.")
        
        observations_b = np.c_[
            observations, np.ones(observations.shape[0])
        ]
        return np.matmul(observations_b, self._weights)

    def save(self, directory: str) -> None:
        """
        Save the model using the Artifact class.

        Args:
            directory (str): The directory where the model should be saved.
        """
        self._artifact = Artifact(
            asset_path=f"{directory}/{self.__class__.__name__}.bin",
            version="1.0.0",
            data=b"",
            metadata={},
            type_="model:regression",
            tags=["multiple_linear_regression"]
        )

        model_data = {
            'parameters': self._parameters
        }
        self._artifact.data = str(model_data).encode()

        self._artifact.save()

    def load(self, directory: str, artifact_id: str) -> None:
        """
        Load the model using the Artifact class.

        Args:
            directory (str): The directory where the model is stored.
            artifact_id (str): The unique ID of the model artifact to be loaded.
        """
        self._artifact = Artifact(
            asset_path=f"{directory}/{artifact_id}.bin",
            version="1.0.0",
            data=b"",
            metadata={},
            type_="model:regression",
            tags=["multiple_linear_regression"]
        )

        loaded_data = self._artifact.read()
        model_data = eval(loaded_data.decode())
        self._parameters = model_data['parameters']
        self._weights = self._parameters.get('weights')
