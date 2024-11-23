from autoop.core.ml.model.classification.decision_tree_classification import (
    DecisionTreeClassificationModel,
)
from autoop.core.ml.model.classification.random_forest import (
    RandomForestClassifierModel,
)
from autoop.core.ml.model.classification.knn import (
    KNearestNeighbors,
)


def get_classification_models() -> dict:
    """
    Retrieves a dictionary of available classification models.

    Returns:
        dict: A dictionary where keys are model names and values are the
            corresponding model classes.
    """
    return {
        "Decision Tree Classification": DecisionTreeClassificationModel,
        "Random Forest": RandomForestClassifierModel,
        "KNN": KNearestNeighbors,
    }
