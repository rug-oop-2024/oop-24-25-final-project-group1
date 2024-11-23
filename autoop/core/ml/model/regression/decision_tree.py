from sklearn.tree import DecisionTreeRegressor
from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
import numpy as np
from pydantic import PrivateAttr, Field


class DecisionTreeRegressionModel(Model):
    """
    Decision Tree Regression model that uses scikit-learn's
    DecisionTreeRegressor to perform regression tasks.

    Attributes:
        _model (DecisionTreeRegressor): The decision tree regression model from
            scikit-learn.
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split
            an internal node.
    """

    _model: DecisionTreeRegressor = PrivateAttr()
    max_depth: int = Field(
        default=None, ge=1, description="Maximum depth of the tree"
    )
    min_samples_split: int = Field(
        default=2,
        ge=2,
        description="Minimum number of samples required to "
                    "split an internal node",
    )

    def __init__(
        self,
        name: str = "test_model",
        asset_path: str = "./tmp",
        version: str = "0.1",
        **data,
    ) -> None:
        """
        Initializes the Decision Tree Regression model with specified
        hyperparameters.

        Args:
            name (str): Name of the model. Defaults to "test_model".
            asset_path (str): Path to store model artifacts.
                Defaults to "./tmp".
            version (str): Version of the model. Defaults to "0.1".
            **data: Additional parameters passed to the model.
        """
        super().__init__(name=name, asset_path=asset_path, version=version, **data)
        self._model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
        )
        self._parameters = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }
        self._type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the Decision Tree Regression model to the provided training data.

        Args:
            observations (np.ndarray): Feature matrix of
                shape (n_samples, p_features).
            ground_truth (np.ndarray): Target vector of shape (n_samples,).
        """
        self._model.fit(observations, ground_truth)
        self._parameters["tree_structure"] = self._model.tree_
        self._parameters["trained"] = True

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
