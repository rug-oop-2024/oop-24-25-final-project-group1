from autoop.core.ml.model.model import Model
"""
This module provides a factory function to retrieve machine learning models by name.

Classes:
    Model: Base class for all models.
    MultipleLinearRegression: Implements multiple linear regression.
    DecisionTreeRegressionModel: Implements decision tree regression.
    LassoRegressionModel: Implements lasso regression.
    KNearestNeighbors: Implements k-nearest neighbors classification.
    DecisionTreeClassificationModel: Implements decision tree classification.
    RandomForestClassifierModel: Implements random forest classification.

Constants:
    REGRESSION_MODELS (list): List of available regression model names.
    CLASSIFICATION_MODELS (list): List of available classification model names.

Functions:
    get_model(model_name: str) -> Model:
        Factory function to get a model by name.
        Args:
            model_name (str): The name of the model to retrieve.
        Returns:
            Model: An instance of the requested model.
        Raises:
            ValueError: If the model name is not recognized.
"""
from autoop.core.ml.model.regression.mulitple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.decision_tree import (
    DecisionTreeRegressionModel,
)
from autoop.core.ml.model.regression.lasso_regression import (
    LassoRegressionModel,
)
from autoop.core.ml.model.classification.knn import (
    KNearestNeighbors,
)
from autoop.core.ml.model.classification.decision_tree_classification import (
    DecisionTreeClassificationModel,
)
from autoop.core.ml.model.classification.random_forest import (
    RandomForestClassifierModel,
)

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "DecisionTreeRegressionModel",
    "LassoRegressionModel",
]

CLASSIFICATION_MODELS = [
    "KNNModel",
    "DecisionTreeClassificationModel",
    "RandomForestModel",
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "MultipleLinearRegression":
        return MultipleLinearRegression()
    elif model_name == "DecisionTreeRegressionModel":
        return DecisionTreeRegressionModel()
    elif model_name == "LassoRegressionModel":
        return LassoRegressionModel()
    elif model_name == "KNNModel":
        return KNearestNeighbors()
    elif model_name == "DecisionTreeClassificationModel":
        return DecisionTreeClassificationModel()
    elif model_name == "RandomForestModel":
        return RandomForestClassifierModel()
    else:
        raise ValueError(f"'{model_name}' is not recognized. "
                         f"Please use a valid model name.")
