import time
import pandas as pd
from src.utils import save_csv
from src.model import dataset_creation, modelling, pf_metrics, weighted_score

if __name__ == "__main__":
    start_time = time.time()

    fpath: str = save_csv()
    X_train, X_test, y_train, y_test = dataset_creation(fpath)
    model, y_pred, y_test = modelling(X_train, X_test, y_train, y_test)

    pf_metrics(y_pred, y_test)
    end_time = time.time()

    print(weighted_score(model, start_time, end_time, X_test, y_test))
