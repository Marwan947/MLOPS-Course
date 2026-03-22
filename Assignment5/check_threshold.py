import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(TRACKING_URI)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking threshold for Run ID: {run_id}")

client = MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("ERROR: 'accuracy' metric not found in MLflow run.")
    sys.exit(1)

THRESHOLD = 0.85
print(f"Accuracy: {accuracy:.4f}, Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
    sys.exit(1)

print(f"PASSED: Accuracy {accuracy:.4f} meets threshold {THRESHOLD}")
