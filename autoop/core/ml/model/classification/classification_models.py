from autoop.core.ml.model.classification.decision_tree_classification import DecisionTreeClassificationModel
from autoop.core.ml.model.classification.random_forest import RandomForestClassifierModel
from autoop.core.ml.model.classification.knn import KNearestNeighbors

def get_classification_models():
    return {
        "Decision Tree Classification": DecisionTreeClassificationModel,
        "Random Forest": RandomForestClassifierModel,
        "KNN": KNearestNeighbors
    }
