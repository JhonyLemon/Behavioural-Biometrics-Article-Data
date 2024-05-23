import os
from typing import Dict, Any, List
import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt


def get_user_ids(results: Dict[int, Any]) -> List[int]:
    return list(results.keys())


def prepare_folder(folder: List[str]) -> str:
    path = os.path.join(os.getcwd(), *folder)
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
    plt.savefig(os.path.join(
        prepare_folder([folder, 'all']),
        'accuracy_comparison.png'
    ))
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
    plt.savefig(os.path.join(
        prepare_folder([folder, 'all']),
        'f_measure_distribution.png'
    ))


def plot_confusion_matrix_heatmaps(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    confusion_matrices = [results[user_id]['confusion_matrix'] for user_id in user_ids]

    average_cm = np.round(np.mean(confusion_matrices, axis=0)).astype(int)

    plt.figure(figsize=(8, 6))
    sns.heatmap(average_cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Average Confusion Matrix Across Users')
    plt.savefig(os.path.join(
        prepare_folder([folder, 'average']),
        'average_confusion_matrix.png'
    ))
    plt.close()

    for user_id, cm in zip(user_ids, confusion_matrices):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for User ID: {user_id}')
        plt.savefig(os.path.join(
            prepare_folder([folder, 'user', f'{user_id}']),
            f'confusion_matrix_{user_id}.png'
        ))
        plt.close()


def plot_precision_recall_curves(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    precision = [[results[user_id]['report']['0']['precision'], results[user_id]['report']['1']['precision']] for
                 user_id in user_ids]
    recall = [[results[user_id]['report']['0']['recall'], results[user_id]['report']['1']['recall']] for user_id in
              user_ids]

    average_precision = np.mean(precision, axis=0)
    average_recall = np.mean(recall, axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(average_precision, average_recall, marker='o')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Average Precision-Recall Curve Across Users')
    plt.grid(True)
    plt.savefig(os.path.join(
        prepare_folder([folder, 'average']),
        'average_precision_recall.png'
    ))
    plt.close()

    for user_id, (precision_values, recall_values) in zip(user_ids, zip(precision, recall)):
        plt.figure(figsize=(8, 6))
        plt.plot(precision_values, recall_values, marker='o')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title(f'Precision-Recall Curve for User ID: {user_id}')
        plt.grid(True)
        plt.savefig(os.path.join(
            prepare_folder([folder, 'user', f'{user_id}']),
            f'precision_recall_{user_id}.png'
        ))
        plt.close()


def plot_roc_curves(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    roc_curve_data = [
        (results[user_id]['fpr'], results[user_id]['tpr'], results[user_id]['thresholds'], results[user_id]['auc']) for
        user_id in user_ids]

    for user_id, (fpr, tpr, thresholds, auc) in zip(user_ids, roc_curve_data):
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve for User ID: {user_id}, AUC = {auc:.2f}')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristics for all users')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(
        prepare_folder([folder, 'all']),
        'roc_curves_all_users.png'
    ))
    plt.close()

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
        plt.savefig(os.path.join(
            prepare_folder([folder, 'user', f'{user_id}']),
            f'roc_curve_{user_id}.png'
        ))
        plt.close()


def plot_feature_importance(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    feature_importance = [results[user_id]['feature_importance'] for user_id in user_ids]
    features = results[user_ids[0]]['features']

    average_importance = np.mean(feature_importance, axis=0) # take mean over all users

    plt.figure(figsize=(10, 6))
    plt.bar(features, average_importance)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Average Feature Importance Across Users')
    plt.xticks(rotation=45, fontsize=10, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(
        prepare_folder([folder, 'average']),
        'average_feature_importance.png'
    ))
    plt.close()

    for user_id, importance in zip(user_ids, feature_importance):
        plt.figure(figsize=(10, 6))
        plt.bar(features, importance)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance for User ID: {user_id}')
        plt.xticks(rotation=45, fontsize=10, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(
            prepare_folder([folder, 'user', f'{user_id}']),
            f'feature_importance_{user_id}.png'
        ))
        plt.close()


def plot_histogram_distribution(results: Dict[int, Any], folder: str):
    user_ids = get_user_ids(results)
    y_pred_prob = [results[user_id]['y_pred_prob'] for user_id in user_ids]

    y_pred_prob_average = [prob for user_id in user_ids for prob in results[user_id]['y_pred_prob']]
    plt.figure()
    plt.hist(y_pred_prob_average, bins=20)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities Across Users')
    plt.savefig(os.path.join(
        prepare_folder([folder, 'average']),
        'average_histogram.png'
    ))
    plt.close()

    for user_id, y_pred_prob_user in zip(user_ids, y_pred_prob):
        plt.figure()
        plt.hist(y_pred_prob_user, bins=20)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Predicted Probabilities for User ID: {user_id}')
        plt.savefig(os.path.join(
            prepare_folder([folder, 'user', f'{user_id}']),
            f'histogram_{user_id}.png'
        ))
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
        plt.savefig(os.path.join(
            prepare_folder([folder, 'user', f'{user_id}']),
            f'learning_curve_{user_id}.png'
        ))
        plt.close()


def plot_boxplot(results: Dict[int, Any], folder: str):
    metrics_results = {
        'accuracy': [results[user_id]['accuracy'] for user_id in get_user_ids(results)],
        'precision': [results[user_id]['report']['weighted avg']['precision'] for user_id in get_user_ids(results)],
        'recall': [results[user_id]['report']['weighted avg']['recall'] for user_id in get_user_ids(results)],
        'f1-score': [results[user_id]['report']['weighted avg']['f1-score'] for user_id in get_user_ids(results)]
    }

    metrics = list(metrics_results.keys())
    values = list(metrics_results.values())
    means = [np.mean(metrics_results[metric]) for metric in metrics]
    stddev = [np.std(metrics_results[metric]) for metric in metrics]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].boxplot(values, vert=False)
    axs[0].set_yticks(range(1, len(metrics)+1))
    axs[0].set_yticklabels(metrics)
    axs[0].set_xlabel('Score')
    axs[0].set_title('Metrics Distribution Across Users')

    axs[1].errorbar(x=means, y=range(1, len(metrics)+1), xerr=stddev, fmt='ro')
    axs[1].set_yticks(range(1, len(metrics)+1))
    axs[1].set_yticklabels(metrics)
    axs[1].set_xlabel('Score')
    axs[1].set_title('Mean and Standard Deviation of Metrics Across Users')

    fig.tight_layout()
    fig.savefig(os.path.join(
        prepare_folder([folder, 'average']),
        'box_plot.png'
    ))
    fig.close()


def plot_all(results: Dict[int, Any], folder: str):
    plot_accuracy_comparison(results, folder)
    plot_f_measure_distribution(results, folder)
    plot_confusion_matrix_heatmaps(results, folder)
    plot_precision_recall_curves(results, folder)
    plot_roc_curves(results, folder)
    plot_feature_importance(results, folder)
    plot_histogram_distribution(results, folder)
    plot_learning_curves(results, folder)
    plot_boxplot(results, folder)
