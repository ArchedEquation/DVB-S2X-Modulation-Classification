import time
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def dataset_creation(data_path: str):
    label_enc = LabelEncoder()
    data = pd.read_csv(data_path)
    data['modulation'] = label_enc.fit_transform(data['modulation_type'])

    X = data[['magnitude', 'phase', 'caf_1', 'caf_2',
              'caf_3', 'caf_4', 'caf_5', 'scd_mean', 'scd_max']]
    y = data['modulation']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def modelling(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    return rf_model, y_pred, y_test


def pf_metrics(y_pred, y_test):
    rf_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)
    report = classification_report(y_test, y_pred, zero_division=1)
    print("\nRandom Forest Model Accuracy: {:.2f}%".format(rf_accuracy * 100))
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print("\nClassification Report for Random Forest:")
    print(report)


def weighted_score(model, start_time, end_time, X_test, y_test, weight_for_accuracy=0.7, weight_for_latency=0.2, weight_for_model_size=0.1):
    try:
        accuracy = model.score(X_test, y_test)
        _ = model.predict(X_test)
        latency = (end_time - start_time) / len(X_test)

        model_pickle = pickle.dumps(model)
        model_size = len(model_pickle)

        max_latency = 0.01
        max_model_size = 1e6

        normalized_latency = latency/max_latency
        normalized_size = model_size/max_model_size

        score = (weight_for_accuracy * accuracy) - (weight_for_latency *
                                                    normalized_latency) - (weight_for_model_size * normalized_size)

        result_string = (
            f"Accuracy: {accuracy:.3f}\n"
            f"Normalized Latency: {
                normalized_latency:.3f} and Latency: {latency}\n"
            f"Normalized Size: {normalized_size:.3f} and Size: {
                model_size} in Bytes\n"
            f"Final Power-Efficient Score: {score:.3f}")

        return result_string

    except Exception as e:
        print(f"Error calculating power-efficient score: {e}")
        return None
