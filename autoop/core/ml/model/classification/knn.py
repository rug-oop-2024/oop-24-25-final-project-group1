import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
from pydantic import PrivateAttr, Field


class KNearestNeighbors(Model):
    """
    K-Nearest Neighbors (KNN) model implementation classifies
    an observation by analyzing the classes of its `k` nearest neighbors.

    Attributes:
        k (int): Number of nearest neighbors to consider.
    """

    k: int = Field(default=3, ge=1, description="Number of neighbors")
    _model: KNeighborsClassifier = PrivateAttr()
    _parameters: dict = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        k: int = 3,
        name: str = "test_model",
        asset_path: str = "./tmp",
        version: str = "0.1",
        **data
    ) -> None:
        """
        Initializes the KNN model with a specified value of "k".

        Args:
            k (int, optional): Number of neighbors. Defaults to 3.
            name (str, optional): Name of the model. Defaults to "test_model".
            asset_path (str, optional): Path to store model artifacts.
                Defaults to "./tmp".
            version (str, optional): Version of the model. Defaults to "0.1".
            **data: Additional data for initialization.
        """
        super().__init__(
            name=name, asset_path=asset_path, version=version, **data
        )
        self.k = k
        self._parameters["k"] = k
        self._type = "classification"
        self._model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the KNN model to the provided training data.

        Args:
            observations (np.ndarray): Feature matrix of
                shape (n_samples, n_features).
            ground_truth (np.ndarray): Target vector of shape (n_samples,).
        """
        X = np.asarray(observations)
        self._model.fit(X, ground_truth)
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth,
            "k": self.k
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the class of each observation.

        Args:
            observations (np.ndarray): Feature matrix of
                shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        return self._model.predict(observations)

    def save(self, directory: str) -> None:
        """
        Save the model using the Artifact class.

        Args:
            directory (str): The directory where the model should be saved.
        """
        model_data = {
            "parameters": self._parameters
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
        self._parameters = model_data["parameters"]
        self.k = self._parameters.get("k", 3)
        self._model = KNeighborsClassifier(n_neighbors=self.k)
        if (
            "observations" in self._parameters
            and "ground_truth" in self._parameters
        ):
            self._model.fit(
                self._parameters["observations"],
                self._parameters["ground_truth"]
            )
