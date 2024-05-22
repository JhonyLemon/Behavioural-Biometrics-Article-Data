from clasification import standardize_columns, train_user_specific_classifiers, classifier_random_forest_classifier, \
    classifier_decision_tree_classifier
from loader import load_users
from features import generate_features, convert_feature_to_dataframe
from plots import plot_all

users = load_users("dataset")
features_generated = generate_features(users)
converted_features = convert_feature_to_dataframe(features_generated)
standardized_features = standardize_columns(converted_features, ['keyboard_dwell', 'keyboard_flight',
                                                                 'mouse_trajectory_distance', 'mouse_trajectory_time',
                                                                 'mouse_dwell', 'mouse_flight'])

results = {}
for user_file_id, user_specific_feature in standardized_features.items():
    results[user_file_id] = train_user_specific_classifiers(
        user_specific_feature,
        user_file_id,
        classifier_decision_tree_classifier
    )

plot_all(results, 'plots')
