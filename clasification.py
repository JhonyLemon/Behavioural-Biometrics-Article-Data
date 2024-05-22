from typing import List, Dict, Callable, Any

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler


def classifier_decision_tree_classifier():
    return DecisionTreeClassifier()


def classifier_random_forest_classifier():
    return RandomForestClassifier(random_state=42)


def standardize_columns(features: Dict[int, pd.DataFrame], columns_to_normalize: List[str]):
    scaler = StandardScaler()
    for user, user_features in features.items():
        user_features[columns_to_normalize] = scaler.fit_transform(user_features[columns_to_normalize])
    return features


def train_user_specific_classifiers(features: pd.DataFrame, user_id: int, classifier: Callable) -> Dict[str, Any]:
    x = features.drop(columns=['valid'])
    y = features['valid'].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    clf = classifier()

    clf.fit(x_train, y_train)

    y_pred_prob = clf.predict_proba(x_test)[:, 1]  # Probability estimates of the positive class

    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    feature_importance = clf.feature_importances_
    features = x.columns

    train_sizes, train_scores, test_scores = learning_curve(clf, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    print(f"User ID: {user_id}")
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    return {
        'classifier': clf,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': auc,
        'feature_importance': feature_importance,
        'features': features,
        'y_pred_prob': y_pred_prob,
        'train_sizes': train_sizes,
        'train_scores_mean': train_scores_mean,
        'test_scores_mean': test_scores_mean
    }
