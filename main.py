import time
from src.DVBS2X import DVBS2X
from src.function import train, test
from sklearn.metrics import accuracy_score, classification_report
from src.utils import weighted_score


start_time = time.time()
if __name__ == "__main__":
    # Invoking the train and test functions.
    rf_model = train()
    X_t, y_t, y_pred = test(rf_model=rf_model)

    # Evaluating the model in terms of performance as well as model latency and size.
    rf_accuracy = accuracy_score(y_t, y_pred)
    print("\nRandom Forest Model Accuracy: {:.2f}%".format(rf_accuracy * 100))
    print("\nClassification Report for Random Forest:")
    print(classification_report(y_t, y_pred))
    end_time = time.time()

print(weighted_score(model=rf_model, start_time=start_time,
      end_time=end_time, X_test=X_t, y_test=y_t))
