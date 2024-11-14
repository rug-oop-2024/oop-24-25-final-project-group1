from autoop.core.ml.model.classification.logistic_regression import LogisticRegressionModel
from autoop.core.ml.model.classification.random_forest import RandomForestClassifierModel
from autoop.core.ml.model.classification.knn import KNearestNeighbors

def get_classification_models():
    return {
        "Logistic Regression": LogisticRegressionModel,
        "Random Forest": RandomForestClassifierModel,
        "KNN": KNearestNeighbors
    }
