from autoop.core.ml.model.model import Model
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
