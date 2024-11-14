import numpy as np
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
from pydantic import PrivateAttr, Field
from typing import Dict


class KNearestNeighbors(Model):
    """
    This K-Nearest Neighbors (KNN) model implementation classifies
    an observation by analyzing the classes of its `k` nearest neighbors.

    Attributes:
    - k (int): The number of nearest neighbors to consider when making predictions.
    """
    k: int = Field(default=3, ge=1, description="Number of neighbors")
    _hyperparameters: Dict = PrivateAttr(default_factory=dict)

    def __init__(self, k=3, name: str = "test_model", asset_path: str = "./tmp", version: str = "0.1", **data):
        """
        Initializes the KNN model with a value of "k".
        """
        super().__init__(name=name, asset_path=asset_path, version=version, **data)
        self.k = k
        self._hyperparameters['k'] = k
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the KNN model to the provided training data.

        Args:
            observations (np.ndarray): Feature matrix of shape (n_samples, n_features).
            ground_truth (np.ndarray): Target vector of shape (n_samples,).
        """
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the class of each observation.

        Args:
            observations (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> int:
        """
        Attributes a label to a single observation based on `k` nearest neighbors.

        Args:
            observation (np.ndarray): Feature vector of shape (n_features,).

        Returns:
            int: Predicted class label.
        """
        distances = np.linalg.norm(self._parameters["observations"] - observation, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self._parameters["ground_truth"][i] for i in k_indices]

        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        most_common_label = max(label_counts, key=label_counts.get)
        return most_common_label

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
        self.k = self._hyperparameters.get('k', 3)
