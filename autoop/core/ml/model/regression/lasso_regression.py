from sklearn.linear_model import Lasso as SKLasso
from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
import numpy as np
from pydantic import PrivateAttr


class LassoRegressionModel(Model):
    """
    Lasso Regression model that performs linear regression using L1
    regularization. It uses scikit-learn's Lasso implementation.

    Attributes:
        _model (SKLasso): The Lasso regression model from scikit-learn.
    """

    _model: SKLasso = PrivateAttr()

    def __init__(
        self,
        alpha: float = 1.0,
        name: str = "lasso_model",
        asset_path: str = "./tmp",
        version: str = "0.1",
        **data,
    ) -> None:
        """
        Initializes the Lasso model with a specified alpha parameter.

        Args:
            alpha (float): Regularization strength (default: 1.0)
            name (str): Name of the model. Defaults to "lasso_model"
            asset_path (str): Path to store model artifacts
                Defaults to "./tmp"
            version (str): Version of the model. Defaults to "0.1"
            **data: Additional parameters for the model
        """
        super().__init__(name=name, asset_path=asset_path, version=version, **data)
        self._model = SKLasso(alpha=alpha)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }
        self._type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the Lasso regression model to the provided training data.

        Args:
            observations (np.ndarray): Feature matrix of
                shape (n_samples, p_features).
            ground_truth (np.ndarray): Target vector of
                shape (n_samples,).
        """
        self._model.fit(observations, ground_truth)
        self._parameters["coefficients"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the values for the provided observations.

        Args:
            observations (np.ndarray): Feature matrix of
                shape (n_samples, p_features).

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
            "model": self._model,
            "parameters": self._parameters,
        }
        self._artifact.data = str(model_data).encode()
        self._artifact.save(directory)

    def load(self, directory: str, artifact_id: str) -> None:
        """
        Load the model using the Artifact class.

        Args:
            directory (str): The directory where the model is stored.
            artifact_id (str): The unique ID of the model
                artifact to be loaded.
        """
        loaded_artifact = Artifact.load(directory, artifact_id)
        model_data = eval(loaded_artifact.data.decode())
        self._model = model_data["model"]
        self._parameters = model_data["parameters"]
        self._type = "regression"
