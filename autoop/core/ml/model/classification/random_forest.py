import numpy as np
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
from pydantic import PrivateAttr, Field


class RandomForestClassifierModel(Model):
    """
    Random Forest Classifier model implementation.

    Uses a RandomForestClassifier from scikit-learn to perform classification tasks.
    """
    n_estimators: int = Field(default=100, ge=1, description="Number of trees in the forest")
    _model: RandomForestClassifier = PrivateAttr()
    _hyperparameters: Dict = PrivateAttr(default_factory=dict)

    def __init__(self, n_estimators: int = 100, max_depth: int = None, **data):
        """
        Initializes the Random Forest Classifier model with specified hyperparameters.

        Args:
            n_estimators (int): The number of trees in the forest (default: 100).
            max_depth (int): The maximum depth of the trees (default: None).
        """
        super().__init__(**data)
        # Set hyperparameters in the hyperparameters dictionary
        self._hyperparameters['n_estimators'] = n_estimators
        self._hyperparameters['max_depth'] = max_depth

        self._model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the Random Forest model to the provided training data.

        Args:
            observations (np.ndarray): Feature matrix of shape (n_samples, n_features).
            ground_truth (np.ndarray): Target vector of shape (n_samples,).
        """
        self._model.fit(observations, ground_truth)
        self._parameters['trained'] = True

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
            artifact_id (str): The unique ID of the model artifact to be loaded.
        """
        loaded_artifact = Artifact.load(directory, artifact_id)
        model_data = eval(loaded_artifact.data.decode())
        self._parameters = model_data['parameters']
        self._hyperparameters = model_data['hyperparameters']
        self._model = RandomForestClassifier(
            n_estimators=self._hyperparameters['n_estimators'],
            max_depth=self._hyperparameters['max_depth']
        )