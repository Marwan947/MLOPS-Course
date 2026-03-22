import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Assignment5_Pipeline")

df = pd.read_csv("data/train.csv")
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
y = df["label"].values

rng = np.random.default_rng(seed=0)
X = X + rng.normal(0, 0.5, X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    mlflow.log_params(params)

    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, name="iris_classifier")

    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {acc:.4f}")

with open("model_info.txt", "w") as f:
    f.write(run_id)

print("Wrote model_info.txt")
