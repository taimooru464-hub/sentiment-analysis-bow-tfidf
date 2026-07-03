"""
evaluation.py
--------------
Shared evaluation helpers so both the BoW and TF-IDF models are scored
the exact same way (accuracy, precision, recall, F1, confusion matrix).
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(y_true, y_pred, pos_label="pos"):
    """
    Computes the standard classification metrics for a set of predictions.
    Returns a dictionary so results from BoW and TF-IDF can be dropped
    straight into a comparison DataFrame.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=pos_label),
        "Recall": recall_score(y_true, y_pred, pos_label=pos_label),
        "F1 Score": f1_score(y_true, y_pred, pos_label=pos_label),
    }


def get_confusion_matrix(y_true, y_pred, labels=("neg", "pos")):
    return confusion_matrix(y_true, y_pred, labels=labels)


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)
