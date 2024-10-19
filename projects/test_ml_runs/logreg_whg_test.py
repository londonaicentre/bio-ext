# log_reg synthetic weight/height vs gender
# test creation of artifacts using MLFlow and Minio

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

def create_dummy_data(n_samples=10000):
    np.random.seed(42)
    
    # generate random heights 
    male_heights = np.random.normal(178, 7, n_samples // 2)
    female_heights = np.random.normal(165, 6, n_samples // 2)
    heights = np.concatenate([male_heights, female_heights])
    
    # generate random weights with correlation to height)
    weights = heights * 0.55 + np.random.normal(0, 5, n_samples)
    
    # create y labels with additional noise
    genders = np.concatenate([np.ones(n_samples // 2), np.zeros(n_samples // 2)])
    random_flip = np.random.random(n_samples) < 0.1  # TRUE where < 0.1 -> 10% chance to flip gender
    genders = np.where(random_flip, 1 - genders, genders)

    # create and shuffle dataset
    df = pd.DataFrame({
        'height': heights,
        'weight': weights,
        'gender': genders
    })
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def mlflow_logistic_regression():
    warnings.filterwarnings("ignore")
    
    mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.set_experiment("test_logreg_w_h_g")

    df = create_dummy_data()
    X = df[['height', 'weight']]
    y = df['gender']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__max_iter': [10, 100, 1000]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')

    with mlflow.start_run() as run:
        grid_search.fit(X_train, y_train)

        mlflow.log_params(grid_search.best_params_)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # log metadata and model
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            best_model, 
            "tuned_model",
            signature=signature
        )

    print(f"MLflow run completed. Run ID: {run.info.run_id}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return run.info.run_id

if __name__ == "__main__":
    try:
        run_id = mlflow_logistic_regression()
    except Exception as e:
        print(f"Failure: {e}")
        exit(1)