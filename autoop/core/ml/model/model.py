from autoop.core.ml.artifact import Artifact
import numpy as np
from typing import Dict, Union, List
from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr
from copy import deepcopy


class Model(BaseModel, ABC):
    """
    Abstract base class for all learning models.

    This class defines the standard structure for machine learning models,
    including methods for fitting, predicting, saving, and loading models.

    Attributes:
        _parameters (Dict): Stores model-specific parameters.
        _artifact (Artifact): Manages the artifact for the model.
        _type (str): The type of the model (e.g., classification, regression).
    """

    _parameters: Dict = PrivateAttr(default_factory=dict)
    _artifact: Artifact = PrivateAttr()
    _type: str = PrivateAttr()

    def __init__(
        self,
        name: str,
        asset_path: str,
        version: str,
        **data: Dict[str, Union[str, int, float, bool, List, dict]],
    ) -> None:
        """
        Initializes the Model with an associated Artifact.

        Args:
            name (str): Name of the model artifact.
            asset_path (str): Path for storing the artifact.
            version (str): Version of the model.
            **data (Dict[str, Union[str, int, float, bool, List, dict]]):
                Additional data passed to the BaseModel initializer.
        """
        super().__init__(**data)
        self._artifact = Artifact(
            name=name,
            asset_path=asset_path,
            version=version,
            data=b"",  # Placeholder for model binary data
            metadata={},
            type="model",
            tags=["machine_learning"]
        )

    @property
    def parameters(self) -> Dict:
        """
        Returns a deepcopy of the model parameters.

        Returns:
            Dict: A deepcopy of the model parameters.
        """
        return deepcopy(self._parameters)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the provided training data.

        Args:
            observations (np.ndarray): Feature matrix of
                shape (n_samples, n_features).
            ground_truth (np.ndarray): Target vector of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the values for the provided observations.

        Args:
            observations (np.ndarray): Feature matrix of
                shape (n_samples, n_features).

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
        model_data = {"parameters": self._parameters}
        self._artifact.data = str(model_data).encode()
        self._artifact.asset_path = f"{directory}/{self._artifact.name}.bin"

    def load(self, directory: str, artifact_id: str) -> None:
        """
        Load the model using the Artifact class.

        Args:
            directory (str): The directory where the model is stored.
            artifact_id (str): Unique ID of the model artifact to be loaded.
        """
        self._artifact.asset_path = f"{directory}/{artifact_id}.bin"
        loaded_data = self._artifact.read()
        model_data = eval(loaded_data.decode())
        self._parameters = model_data.get("parameters", {})

    def set_params(
        self, **params: Dict[str, Union[str, int, float, bool, List, dict]]
    ) -> None:
        """
        Set the model parameters.

        Args:
            **params (Dict[str, Union[str, int, float, bool, List, dict]]):
                Arbitrary keyword arguments representing model parameters.
        """
        for key, value in params.items():
            self._parameters[key] = value

    def get_params(self) -> Dict:
        """
        Get the model parameters.

        Returns:
            Dict: The current model parameters.
        """
        return self.parameters

    def is_trained(self) -> bool:
        """
        Check if the model has been trained.

        Returns:
            bool: True if the model has been trained, False otherwise.
        """
        return bool(self._parameters)

    @property
    def type(self) -> str:
        """
        Get the type of the model.

        Returns:
            str: The type of the model.
        """
        return self._type
