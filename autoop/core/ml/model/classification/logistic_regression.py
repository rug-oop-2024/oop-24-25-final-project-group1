import numpy as np
from sklearn.linear_model import LogisticRegression
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
from pydantic import PrivateAttr


class LogisticRegressionModel(Model):
    """
    Logistic Regression model implementation that inherits from the base
    "Model" class.

    Uses LogisticRegression from scikit-learn to perform classification tasks.
    """
    _model: LogisticRegression = PrivateAttr()
    _parameters: dict = PrivateAttr(default_factory=dict)

    def __init__(self, name: str = "test_model", asset_path: str = "./tmp", version: str = "0.1", max_iter: int = 100, penalty: str = 'l2', C: float = 1.0, **data):
        """
        Initializes the Logistic Regression model with specified parameters.

        Args:
            max_iter (int): Maximum number of iterations for the solver (default: 100).
            penalty (str): The norm used in the penalization ('l2' by default).
            C (float): Inverse of regularization strength (default: 1.0).
        """
        super().__init__(name=name, asset_path=asset_path, version=version, **data)
        self._parameters = {
            "max_iter": max_iter,
            "penalty": penalty,
            "C": C
        }
        self._type = "classification"
        self._model = LogisticRegression(max_iter=max_iter, penalty=penalty, C=C)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the Logistic Regression model to the provided training data.

        Args:
            observations (np.ndarray): Feature matrix of shape (n_samples, n_features).
            ground_truth (np.ndarray): Target vector of shape (n_samples,).
        """
        X = np.asarray(observations)
        y = np.asarray(ground_truth).ravel()
        self._model.fit(X, y)
        self._parameters.update({
            "observations": observations,
            "ground_truth": ground_truth
        })

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the values for the provided observations.

        Args:
            observations (np.ndarray): Feature matrix of shape (n_samples, n_features).

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
            'parameters': self._parameters
        }
        self._artifact.data = str(model_data).encode()
        self._artifact.save(directory)

    def load(self, directory: str, artifact_id: str) -> None:
        """
        Load the model using the Artifact class.

        Args:
            directory (str): The directory where the model is stored.
            artifact_id (str): The unique ID of the model artifact to be loaded.
        """
        loaded_artifact = Artifact.load(directory, artifact_id)
        model_data = eval(loaded_artifact.data.decode())
        self._parameters = model_data['parameters']

        # Reinitialize the model with the saved parameters
        self._model = LogisticRegression(
            max_iter=self._parameters.get('max_iter', 100),
            penalty=self._parameters.get('penalty', 'l2'),
            C=self._parameters.get('C', 1.0)
        )
        if "observations" in self._parameters and "ground_truth" in self._parameters:
            self._model.fit(self._parameters["observations"], self._parameters["ground_truth"])
