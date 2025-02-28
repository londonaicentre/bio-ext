# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# This script is adapted from the test script by https://github.com/sachua/mlflow-docker-compose

import sys
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

load_dotenv(override=True)


def eval_metrics(actual, pred):
    rmse = np.sqrt(root_mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("setting mlflow tracking url")
    mlflow.set_tracking_uri("http://bioextmlflow:5001")

    np.random.seed(40)

    data = pd.read_csv("data/wine-quality.csv")

    print("train test split")
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    status = requests.get("http://bioextmlflow:5001", timeout=5).status_code
    print(status)

    print("starting run")
    with mlflow.start_run():
        print("training")
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        print("predicting")
        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
