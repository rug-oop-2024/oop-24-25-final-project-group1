from abc import ABC, abstractmethod
from typing import Any
import numpy as np


def get_metric(name: str):
    """
    Factory function to get a metric by name.
    Args:
        name (str): The name of the metric to retrieve.

    Returns:
        Metric: An instance of a metric class corresponding to the name.
    """
    if name == "Mean Squared Error":
        return MeanSquaredError()
    elif name == "Accuracy":
        return Accuracy()
    elif name == "Balanced Accuracy":
        return BalancedAccuracy()
    elif name == "Macro Precision":
        return MacroPrecision()
    elif name == "R² Score":
        return R2Score()
    elif name == "Mean Absolute Error":
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
    
    @abstractmethod
    def __str__(self):
        """
        Return a string representation of the metric.

        Returns:
            str: The string representation of the metric.
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
    
    def __str__(self):
        return "Mean Squared Error"

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
    
    def __str__(self):
        return "Accuracy"

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
    
    def __str__(self):
        return "Balanced Accuracy"


class LogLoss(Metric):
    """
    Implementation of Log Loss (Cross-Entropy Loss) for multi-class classification tasks.
    """
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate Log Loss for multi-class classification.

        Args:
            ground_truth (np.ndarray): The true class labels (shape: n_samples,).
            prediction (np.ndarray): The predicted probabilities for each class
                                     (shape: n_samples, n_classes).

        Returns:
            float: The computed Log Loss value.
        """
        prediction = np.clip(prediction, 1e-15, 1 - 1e-15)  # Avoid log(0) issues
        n_samples = ground_truth.shape[0]
        
        n_classes = prediction.shape[1]
        one_hot_ground_truth = np.zeros((n_samples, n_classes))
        one_hot_ground_truth[np.arange(n_samples), ground_truth] = 1

        log_loss = -np.sum(one_hot_ground_truth * np.log(prediction)) / n_samples
        return log_loss

    def __str__(self):
        return "Log Loss"
    
    
class MacroPrecision(Metric):
    """Class for MacroPrecision. Inherits from Metric
    """

    def __call__(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
        """Calculates the Macro Precision using the __call__ method.

        Args:
            true_labels (np.ndarray): True Values
            predicted_labels (np.ndarray): Predicted Values

        Returns:
            float: Calculated Macro Precision
        """
        classes = np.unique(true_labels)
        precision_values = [
            self._calculate_precision_for_class(true_labels, predicted_labels, label)
            for label in classes
        ]

        return np.mean(precision_values)

    def _calculate_precision_for_class(self, true_labels, predicted_labels, label):
        """Helper function to calculate precision for a specific class.

        Args:
            true_labels (np.ndarray): True Values
            predicted_labels (np.ndarray): Predicted Values
            label: The class label to calculate precision for

        Returns:
            float: Precision value for the specified class
        """
        true_positives = np.sum((predicted_labels == label) & (true_labels == label))
        total_predicted = np.sum(predicted_labels == label)

        return true_positives / total_predicted if total_predicted > 0 else 0.0
    
    def __str__(self):
        return "Macro Precision"

    

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
    
    def __str__(self):
        return "R² Score"

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
    
    def __str__(self):
        return "Mean Absolute Error"


def get_regression_metrics() -> list[Metric]:
    """
    Get a list of regression metrics.

    Returns:
        List[Metric]: A list of regression metrics.
    """
    return [MeanSquaredError(), R2Score(), MeanAbsoluteError()]


def get_classification_metrics() -> list[Metric]:
    """
    Get a list of classification metrics.

    Returns:
        List[Metric]: A list of classification metrics.
    """
    return [Accuracy(), BalancedAccuracy(), MacroPrecision()]