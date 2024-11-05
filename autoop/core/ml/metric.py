from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "balanced_accuracy",
    "cohens_kappa",
    "r2_score",
    "mean_absolute_error",
]

def get_metric(name: str):
    """
    Factory function to get a metric by name.
    Args:
        name (str): The name of the metric to retrieve.

    Returns:
        Metric: An instance of a metric class corresponding to the name.
    """
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "balanced_accuracy":
        return BalancedAccuracy()
    elif name == "cohens_kappa":
        return CohensKappa()
    elif name == "r2_score":
        return R2Score()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    else:
        raise ValueError(f"Unknown metric: {name}")

class Metric(ABC):
    """
    Base class for all metrics.
    Metrics take ground truth and predictions as input and return a real number.
    """

    @abstractmethod
    def __call__(self, ground_truth: Any, prediction: Any) -> float:
        """
        Calculate the metric value given the ground truth and predictions.

        Args:
            ground_truth (Any): The true values.
            prediction (Any): The predicted values.

        Returns:
            float: The computed metric value.
        """
        pass
    
    def evaluate(self, ground_truth: Any, prediction: Any) -> float:
        """
        Evaluate the metric by calculating the value and printing a summary.

        Args:
            ground_truth (Any): The true values.
            prediction (Any): The predicted values.

        Returns:
            float: The computed metric value.
        """
        result = self.__call__(ground_truth, prediction)
        print(f"Evaluating {self.__class__.__name__}: {result:.4f}")
        return result
        

class MeanSquaredError(Metric):
    """
    Implementation of Mean Squared Error (MSE) metric.
    """
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error between ground truth and predictions.

        Args:
            ground_truth (np.ndarray): The true values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed Mean Squared Error.
        """
        error = np.mean((ground_truth - prediction) ** 2)
        return error

class Accuracy(Metric):
    """
    Implementation of Accuracy metric for multi-class classification tasks.
    """
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate the accuracy by comparing ground truth and predictions.

        Args:
            ground_truth (np.ndarray): The true values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed accuracy value.
        """
        correct_predictions = np.sum(ground_truth == prediction)
        accuracy = correct_predictions / len(ground_truth)
        return accuracy

class BalancedAccuracy(Metric):
    """
    Implementation of Balanced Accuracy metric for classification tasks.
    """
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate the Balanced Accuracy by averaging the recall for each class.

        Args:
            ground_truth (np.ndarray): The true values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed Balanced Accuracy value.
        """
        unique_classes = np.unique(ground_truth)
        recalls = []
        for cls in unique_classes:
            true_positive = np.sum((prediction == cls) & (ground_truth == cls))
            false_negative = np.sum((prediction != cls) & (ground_truth == cls))
            if true_positive + false_negative == 0:
                recalls.append(0.0)
            else:
                recalls.append(true_positive / (true_positive + false_negative))
        balanced_accuracy = np.mean(recalls)
        return balanced_accuracy



class CohensKappa(Metric):
    """
    Implementation of Cohen's Kappa metric for multi-class classification tasks.
    """
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate Cohen's Kappa to measure the agreement between predicted and true values, accounting for chance agreement.

        Args:
            ground_truth (np.ndarray): The true values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed Cohen's Kappa value.
        """
        num_classes = len(np.unique(ground_truth))
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        for true, pred in zip(ground_truth, prediction):
            confusion_matrix[true, pred] += 1
        
        total_samples = np.sum(confusion_matrix)
        observed_accuracy = np.trace(confusion_matrix) / total_samples
        
        row_marginals = np.sum(confusion_matrix, axis=1)
        col_marginals = np.sum(confusion_matrix, axis=0)
        expected_accuracy = np.sum((row_marginals * col_marginals) / total_samples) / total_samples
        
        if expected_accuracy == 1:
            return 1.0 
        
        cohen_kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return cohen_kappa


class R2Score(Metric):
    """
    Implementation of R-squared (R²) metric for regression tasks.
    """
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate the R-squared (R²) value between ground truth and predictions.

        Args:
            ground_truth (np.ndarray): The true values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed R-squared value.
        """
        ss_total = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        ss_residual = np.sum((ground_truth - prediction) ** 2)
        r2_score = 1 - (ss_residual / ss_total)
        return r2_score

class MeanAbsoluteError(Metric):
    """
    Implementation of Mean Absolute Error (MAE) metric for regression tasks.
    """
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Error between ground truth and predictions.

        Args:
            ground_truth (np.ndarray): The true values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The computed Mean Absolute Error.
        """
        error = np.mean(np.abs(ground_truth - prediction))
        return error
