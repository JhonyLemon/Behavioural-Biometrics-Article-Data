import os
from typing import Dict, Any

import seaborn as sns

from matplotlib import pyplot as plt


def get_user_ids(results: Dict[int, Any]):
    return list(results.keys())


def prepare_folder(folder: str):
    path = os.path.join(os.getcwd(), folder)
    os.makedirs(path, exist_ok=True)
    return path


def plot_accuracy_comparison(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    accuracies = [results[user_id]['accuracy'] for user_id in user_ids]
    plt.figure(figsize=(10, 6))
    plt.plot(user_ids, accuracies, linewidth=3)
    plt.xlabel('User ID')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison by User')
    plt.grid(True)
    plt.savefig(os.path.join(prepare_folder(folder), 'accuracy_comparison.png'))
    plt.close()


def plot_f_measure_distribution(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    f1_scores = [results[user_id]['report']['weighted avg']['f1-score'] for user_id in user_ids]
    plt.figure(figsize=(10, 6))
    plt.plot(user_ids, f1_scores, linewidth=3)
    plt.xlabel('User ID')
    plt.ylabel('F1-Score')
    plt.title('F1-Score Comparison by User')
    plt.grid(True)
    plt.savefig(os.path.join(prepare_folder(folder), 'f1_score_comparison.png'))


def plot_confusion_matrix_heatmaps(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    confusion_matrices = [results[user_id]['confusion_matrix'] for user_id in user_ids]
    for user_id, cm in zip(user_ids, confusion_matrices):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for User ID: {user_id}')
        plt.savefig(os.path.join(prepare_folder(folder), f'confusion_matrix_{user_id}.png'))
        plt.close()


def plot_precision_recall_curves(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    precision = [[results[user_id]['report']['0']['precision'], results[user_id]['report']['1']['precision']] for
                 user_id in user_ids]
    recall = [[results[user_id]['report']['0']['recall'], results[user_id]['report']['1']['recall']] for user_id in
              user_ids]
    for user_id, (precision_values, recall_values) in zip(user_ids, zip(precision, recall)):
        plt.figure(figsize=(8, 6))
        plt.plot(precision_values, recall_values, marker='o')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title(f'Precision-Recall Curve for User ID: {user_id}')
        plt.grid(True)
        plt.savefig(os.path.join(prepare_folder(folder), f'precision_recall_curve_{user_id}.png'))
        plt.close()


def plot_roc_curves(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    roc_curve_data = [
        (results[user_id]['fpr'], results[user_id]['tpr'], results[user_id]['thresholds'], results[user_id]['auc']) for
        user_id in user_ids]
    for user_id, (fpr, tpr, thresholds, auc) in zip(user_ids, roc_curve_data):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for User ID: {user_id}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(prepare_folder(folder), f'roc_curve_{user_id}.png'))
        plt.close()


def plot_feature_importance(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    feature_importance = [results[user_id]['feature_importance'] for user_id in user_ids]
    features = results[user_ids[0]]['features']
    for user_id, importance in zip(user_ids, feature_importance):
        plt.figure(figsize=(10, 6))
        plt.bar(features, importance)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance for User ID: {user_id}')
        plt.xticks(rotation=30)
        plt.savefig(os.path.join(prepare_folder(folder), f'feature_importance_{user_id}.png'))
        plt.close()


def plot_histogram_distribution(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    y_pred_prob = [results[user_id]['y_pred_prob'] for user_id in user_ids]
    for user_id, y_pred_prob_user in zip(user_ids, y_pred_prob):
        plt.figure()
        plt.hist(y_pred_prob_user, bins=20)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Predicted Probabilities for User ID: {user_id}')
        plt.savefig(os.path.join(prepare_folder(folder), f'predicted_probability_distribution_{user_id}.png'))
        plt.close()


def plot_learning_curves(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    train_sizes = [results[user_id]['train_sizes'] for user_id in user_ids]
    train_scores_mean = [results[user_id]['train_scores_mean'] for user_id in user_ids]
    test_scores_mean = [results[user_id]['test_scores_mean'] for user_id in user_ids]
    for user_id, train_size, train_score, test_score in zip(user_ids, train_sizes, train_scores_mean, test_scores_mean):
        plt.figure()
        plt.plot(train_size, train_score, 'o-', color="r", label="Training score")
        plt.plot(train_size, test_score, 'o-', color="g", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title(f'Learning Curve for User ID: {user_id}')
        plt.legend(loc="best")
        plt.savefig(os.path.join(prepare_folder(folder), f'learning_curve_{user_id}.png'))
        plt.close()


def plot_all(results: Dict[int, Any], folder: str):
    plot_accuracy_comparison(results, folder)
    plot_f_measure_distribution(results, folder)
    plot_confusion_matrix_heatmaps(results, folder)
    plot_precision_recall_curves(results, folder)
    plot_roc_curves(results, folder)
    plot_feature_importance(results, folder)
    plot_histogram_distribution(results, folder)
    plot_learning_curves(results, folder)
