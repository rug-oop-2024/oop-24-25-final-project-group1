from autoop.core.ml.artifact import Artifact
import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr

class Model(BaseModel, ABC):
    """
    Abstract base class for all learning models.

    Contains the standard fit and predict methods, and can use Artifact-like functionality
    by internally managing an Artifact instance.

    _parameters (Dict): a dictionary that stores model-specific parameters.
    _hyperparameters (Dict): a dictionary that stores model-specific hyperparameters.
    """

    _parameters: Dict = PrivateAttr(default_factory=dict)
    _hyperparameters: Dict = PrivateAttr(default_factory=dict)
    _artifact: Artifact = PrivateAttr()

    def __init__(self, name: str, asset_path: str, version: str, **data):
        super().__init__(**data)
        self._artifact = Artifact(
            name=name,
            asset_path=asset_path,
            version=version,
            data=b"",  # This would be model binary data when saved
            metadata={},
            type="model",
            tags=["machine_learning"]
        )

    @property
    def parameters(self) -> Dict:
        """
        Returns a copy of the model parameters.
        """
        return self._parameters.copy()

    @property
    def hyperparameters(self) -> Dict:
        """
        Returns a copy of the model hyperparameters.
        """
        return self._hyperparameters.copy()

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the provided training data.

        Args:
            observations (np.ndarray): Feature matrix of shape (n_samples, n_features).
            ground_truth (np.ndarray): Target vector of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the values for the provided observations.

        Args:
            observations (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        pass

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
        self._artifact.asset_path = f"{directory}/{self._artifact.name}.bin"

    def load(self, directory: str, artifact_id: str) -> None:
        """
        Load the model using the Artifact class.

        Args:
            directory (str): The directory where the model is stored.
            artifact_id (str): The unique ID of the model artifact to be loaded.
        """
        self._artifact.asset_path = f"{directory}/{artifact_id}.bin"
        loaded_data = self._artifact.read()
        model_data = eval(loaded_data.decode())
        self._parameters = model_data.get('parameters', {})
        self._hyperparameters = model_data.get('hyperparameters', {})

    def set_params(self, **params: Any) -> None:
        """
        Set the model parameters.

        Args:
            **params (Any): Arbitrary keyword arguments representing model parameters.
        """
        for key, value in params.items():
            self._parameters[key] = value

    def set_hyperparams(self, **hyperparams: Any) -> None:
        """
        Set the model hyperparameters.

        Args:
            **hyperparams (Any): Arbitrary keyword arguments representing hyperparameters.
        """
        for key, value in hyperparams.items():
            self._hyperparameters[key] = value

    def get_params(self) -> Dict:
        """
        Get the model parameters.

        Returns:
            Dict: The current model parameters.
        """
        return self.parameters

    def get_hyperparams(self) -> Dict:
        """
        Get the model hyperparameters.

        Returns:
            Dict: The current model hyperparameters.
        """
        return self.hyperparameters

    def is_trained(self) -> bool:
        """
        Check if the model has been trained.

        Returns:
            bool: True if the model has been trained, False otherwise.
        """
        return bool(self._parameters)
