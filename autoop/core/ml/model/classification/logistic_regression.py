import numpy as np
from typing import Dict
from sklearn.linear_model import LogisticRegression
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
from pydantic import PrivateAttr


class LogisticRegressionModel(Model):
    """
    Logistic Regression model implementation that inherits from the base
    "Model" class.

    Uses LogisticRegression from scikit-learn to perform classification
    tasks.
    """
    _model: LogisticRegression = PrivateAttr()
    _hyperparameters: Dict = PrivateAttr(default_factory=dict)

    def __init__(self, max_iter: int = 100, penalty: str = 'l2', C: float = 1.0, **data):
        """
        Initializes the Logistic Regression model with specified hyperparameters.

        Args:
            max_iter (int): Maximum number of iterations for the solver
            (default: 100).
            penalty (str): The norm used in the penalization ('l2' by default).
            C (float): Inverse of regularization strength (default: 1.0).
        """
        super().__init__(**data)
        self._hyperparameters['max_iter'] = max_iter
        self._hyperparameters['penalty'] = penalty
        self._hyperparameters['C'] = C

        self._model = LogisticRegression(max_iter=max_iter, penalty=penalty, C=C)
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the Logistic Regression model to the provided training data.

        Args:
            observations (np.ndarray): Feature matrix of shape
            (n_samples, n_features).
            ground_truth (np.ndarray): Target vector of shape (n_samples,).
        """
        self._model.fit(observations, ground_truth)
        self._parameters['trained'] = True

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the values for the provided observations.

        Args:
            observations (np.ndarray): Feature matrix of shape
            (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        return self._model.predict(observations)

    def save(self, directory: str) -> None:
        """
        Save the model using the Artifact class.

        Args:
            directory (str): The directory where the model should be saved.
        """
        model_data = {
            'parameters': self._parameters,
            'hyperparameters': self._hyperparameters
        }
        self._artifact.data = str(model_data).encode()
        self._artifact.save(directory)

    def load(self, directory: str, artifact_id: str) -> None:
        """
        Load the model using the Artifact class.

        Args:
            directory (str): The directory where the model is stored.
            artifact_id (str): The unique ID of the model artifact to be
            loaded.
        """
        loaded_artifact = Artifact.load(directory, artifact_id)
        model_data = eval(loaded_artifact.data.decode())
        self._parameters = model_data['parameters']
        self._hyperparameters = model_data['hyperparameters']
        self._model = LogisticRegression(
            max_iter=self._hyperparameters['max_iter'],
            penalty=self._hyperparameters['penalty'],
            C=self._hyperparameters['C']
        )
        
    @property
    def type(self) -> str:
        """
        Returns the type of the model.

        Returns:
            str: The type of the model.
        """
        return self._type
