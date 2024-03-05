# Jude Tear
# Evaluation File

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from visualize_data import plot_confusion_matrix

def calculate_specificity_sensitivity_f1(y_true, y_pred):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    class_names = ['Class 0', 'Class 1', 'Class 2']
    plot_confusion_matrix(cm, class_names)

    sensitivity = {}
    specificity = {}
    f1_scores = {}

    total_sum = cm.sum()

    # Calculate metrics for each class
    for i in range(len(cm)):
        # True Positives (TP) are the diagonal elements
        TP = cm[i, i]
        # False Negatives (FN) are the sum of the ith row excluding TP
        FN = sum(cm[i, :]) - TP
        # False Positives (FP) are the sum of the ith column excluding TP
        FP = sum(cm[:, i]) - TP
        # True Negatives (TN) are calculated by excluding the row and column of the current class
        TN = total_sum - (FP + FN + TP)

        # Calculate sensitivity (recall) for the current class
        sensitivity[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
        # Calculate specificity for the current class
        specificity[i] = TN / (TN + FP) if (TN + FP) != 0 else 0
        # Calculate precision for the current class
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        # Calculate F1 score for the current class
        f1_scores[i] = 2 * (precision * sensitivity[i]) / (precision + sensitivity[i]) if (precision + sensitivity[i]) != 0 else 0

    return specificity, sensitivity, f1_scores

def evaluate_model_performance(model, X_test, y_test):
    # Predict classes with the model
    y_pred = model.predict(X_test)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)


    # Calculate metrics
    specificity, sensitivity, f1_score = calculate_specificity_sensitivity_f1(y_true_classes, y_pred_classes)

    # Calculate incorrect indices
    incorrect_indices = np.where(y_true_classes != y_pred_classes)[0]

    return specificity, sensitivity, f1_score, incorrect_indices, y_pred_classes, y_true_classes


def evaluate_model_performance_binary(model, X_test, y_test):
    # Generate predictions
    y_pred_probs = model.predict(X_test)
    
    # Convert predicted probabilities to binary predictions
    y_pred = np.argmax(y_pred_probs, axis=-1)

    # Convert y_test to binary format
    y_test_binary = np.argmax(y_test, axis=-1)

     # Identify the incorrect indices
    incorrect_indices = np.where(y_test_binary != y_pred)[0]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test_binary, y_pred)
    # True Positives
    TP = cm[1, 1]
    # True Negatives
    TN = cm[0, 0]
    # False Positives
    FP = cm[0, 1]
    # False Negatives
    FN = cm[1, 0]

    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity = TP / (TP + FN)
    # Specificity or true negative rate
    specificity = TN / (TN + FP)
    # F1 score
    f1 = f1_score(y_test_binary, y_pred)

    class_names = ['Class 0', 'Class 1']
    plot_confusion_matrix(cm, class_names)

    return specificity, sensitivity, f1, incorrect_indices, y_pred, y_test_binary