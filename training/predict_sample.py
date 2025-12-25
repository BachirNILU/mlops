import os

import mlflow
import pandas as pd

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Local MinIO defaults (training and client-side reads from the artifact store)
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

model_uri = os.environ.get("MODEL_URI", "models:/iris-fastai/Production")
model = mlflow.pyfunc.load_model(model_uri)

sample = pd.DataFrame(
    [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
    ],
    columns=FEATURES,
)

print("Tracking URI:", tracking_uri)
print("Model URI:", model_uri)
print(model.predict(sample))
