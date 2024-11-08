from autoop.core.ml.model.classification.logistic_regression import LogisticRegression
from autoop.core.ml.model.classification.random_forest import RandomForestClassifier
from autoop.core.ml.model.classification.knn import KNearestNeighbors

def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "KNN": KNearestNeighbors
    }
