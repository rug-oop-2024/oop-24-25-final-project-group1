from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd
import io

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects the feature types (categorical or numerical) from a dataset.

    Args:
        dataset: Dataset - an instance of Dataset containing the data.

    Returns:
        List[Feature]: A list of Feature objects with their detected types (categorical or numerical).
    """
    
    if dataset._data is None:
        raise ValueError("The Dataset object does not contain any valid data. Please ensure that the dataset is properly initialized with data.")

    if not isinstance(dataset._data, (str, bytes)):
        raise TypeError("The Dataset object's data should be of type str or bytes.")
    
    csv_data = dataset._data.decode("utf-8")
    data = pd.read_csv(io.StringIO(csv_data))
    features = []

    for column in data.columns:
        feature_type = "categorical" if data[column].dtype == "object" else "numerical"
        features.append(Feature(name=column, type=feature_type))
    
    return features
