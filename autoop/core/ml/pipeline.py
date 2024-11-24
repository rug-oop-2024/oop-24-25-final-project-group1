import pickle
import numpy as np

from typing import List
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features


class Pipeline:
    """
    A class representing a machine learning pipeline. The pipeline handles
    dataset preprocessing, model training, evaluation, and artifact
    management.

    Attributes:
        _dataset (Dataset): The dataset used in the pipeline.
        _model (Model): The model used in the pipeline.
        _input_features (List[Feature]): List of input features.
        _target_feature (Feature): The target feature.
        _metrics (List[Metric]): List of evaluation metrics.
        _split (float): The train-test split ratio.
        _artifacts (dict): Dictionary of artifacts generated during execution.
    """

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split: float = 0.8,
    ) -> None:
        """
        Initializes the pipeline.

        Args:
            metrics (List[Metric]): List of evaluation metrics.
            dataset (Dataset): The dataset to be used.
            model (Model): The model to be used.
            input_features (List[Feature]): List of input features.
            target_feature (Feature): The target feature.
            split (float, optional): Train-test split ratio. Defaults to 0.8.

        Raises:
            ValueError: If target feature type and model type are incompatible.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (
            target_feature.type == "categorical"
            and model.type != "classification"
        ):
            raise ValueError(
                "Model type must be classification for "
                "categorical target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """
        Returns a string representation of the pipeline.

        Returns:
            str: A formatted string describing the pipeline.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Returns the model used in the pipeline.

        Returns:
            Model: The model instance.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Retrieves the artifacts generated during the pipeline execution.

        Returns:
            List[Artifact]: List of artifacts.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """
        Registers an artifact in the pipeline.

        Args:
            name (str): The name of the artifact.
            artifact: The artifact object.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses input and target features, registers artifacts, and
        prepares data for training and testing.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """
        Splits the data into training and
        testing sets based on the split ratio.
        """
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Compacts a list of vectors into a single array by concatenating them.

        Args:
            vectors (List[np.array]): List of numpy arrays.

        Returns:
            np.array: A single concatenated array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model using the training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluates the model using the testing data and calculates metrics.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> dict:
        """
        Executes the pipeline, including preprocessing, splitting, training,
        and evaluation. Also calculates training metrics.

        Returns:
            dict: A dictionary containing training metrics, evaluation metrics,
                and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()

        train_X = self._compact_vectors(self._train_X)
        train_Y = self._train_y
        train_metrics_results = []
        train_predictions = self._model.predict(train_X)
        for metric in self._metrics:
            train_result = metric.evaluate(train_predictions, train_Y)
            train_metrics_results.append((metric, train_result))

        return {
            "training_metrics": train_metrics_results,
            "evaluation_metrics": self._metrics_results,
            "predictions": self._predictions,
        }
