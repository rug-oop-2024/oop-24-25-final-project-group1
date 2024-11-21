from autoop.core.ml.model.regression.mulitple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.decision_tree import DecisionTreeRegressionModel
from autoop.core.ml.model.regression.lasso_regression import LassoRegressionModel


def get_regression_models():
    return {
        "Linear Regression": MultipleLinearRegression,
        "Random Forest": DecisionTreeRegressionModel,
        "Lasso regression": LassoRegressionModel
    }
