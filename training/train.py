from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastai.tabular.all import (
    Categorify,
    FillMissing,
    Normalize,
    RandomSplitter,
    TabularPandas,
    accuracy,
    load_learner,
    range_of,
    set_seed,
    tabular_learner,
)
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


def set_local_minio_defaults() -> None:
    # Training runs on your host machine, so MinIO is localhost:9000 there.
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def load_iris_df() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    rename = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }
    df = df.rename(columns=rename)

    target_names = list(iris.target_names)
    df[TARGET] = df["target"].apply(lambda i: target_names[int(i)])
    df = df.drop(columns=["target"])
    return df


def train_fastai_tabular(df: pd.DataFrame, seed: int, epochs: int, lr: float, bs: int):
    set_seed(seed, reproducible=True)

    splits = RandomSplitter(seed=seed)(range_of(df))
    to = TabularPandas(
        df,
        procs=[Categorify, FillMissing, Normalize],
        cont_names=FEATURES,
        y_names=TARGET,
        splits=splits,
    )
    dls = to.dataloaders(bs=bs)

    learn = tabular_learner(dls, layers=[32, 16], metrics=accuracy)
    learn.fit_one_cycle(epochs, lr)

    val_loss, val_acc = learn.validate()
    return learn, float(val_loss), float(val_acc)


def valid_preds_and_targets(learn):
    preds, targs = learn.get_preds(dl=learn.dls.valid)
    probs = preds.cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    targ_idx = targs.cpu().numpy()
    class_names = list(learn.dls.vocab)
    return probs, pred_idx, targ_idx, class_names


class IrisFastaiPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.learn = None
        self.class_names: list[str] = []

    def load_context(self, context):
        self.learn = load_learner(context.artifacts["learner"])
        self.class_names = list(self.learn.dls.vocab)

    def predict(self, context, model_input):
        if self.learn is None:
            raise RuntimeError("Model not loaded. load_context was not called.")

        if isinstance(model_input, pd.DataFrame):
            df = model_input.copy()
        else:
            df = pd.DataFrame(model_input, columns=self.feature_names)

        df = df[self.feature_names]

        dl = self.learn.dls.test_dl(df)
        preds, _ = self.learn.get_preds(dl=dl)
        probs = preds.cpu().numpy()

        pred_idx = probs.argmax(axis=1)
        pred_labels = [self.class_names[i] for i in pred_idx]

        out = pd.DataFrame({"prediction": pred_labels})
        for i, cls in enumerate(self.class_names):
            out[f"prob_{cls}"] = probs[:, i]
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="iris")
    parser.add_argument("--model-name", default="iris-fastai")
    parser.add_argument("--stage", default="Production")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_local_minio_defaults()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    df = load_iris_df()
    learn, val_loss, val_acc = train_fastai_tabular(
        df=df, seed=args.seed, epochs=args.epochs, lr=args.lr, bs=args.bs
    )

    artifacts_dir = Path("training") / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    learner_path = artifacts_dir / "learner.pkl"
    learn.export(learner_path)

    probs, pred_idx, targ_idx, class_names = valid_preds_and_targets(learn)
    report = classification_report(targ_idx, pred_idx, target_names=class_names)
    report_path = artifacts_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    input_example = df[FEATURES].iloc[:3].copy()

    tmp_wrapper = IrisFastaiPyFuncModel(FEATURES)
    tmp_wrapper.learn = learn
    tmp_wrapper.class_names = class_names
    output_example = tmp_wrapper.predict(None, input_example)

    signature = infer_signature(input_example, output_example)

    pip_reqs = [
        "mlflow==3.8.0",
        "pandas==2.3.3",
        "fastai==2.8.6",
        "torch==2.9.1",
        "torchvision==0.24.1",
        "scikit-learn==1.8.0",
        "boto3==1.42.16",
        # numpy comes transitively, but listing it helps reproducibility
        f"numpy=={np.__version__}",
    ]

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "lr": args.lr,
                "bs": args.bs,
                "seed": args.seed,
            }
        )
        mlflow.log_metrics({"val_loss": val_loss, "val_accuracy": val_acc})

        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(learner_path), artifact_path="fastai_export")

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=IrisFastaiPyFuncModel(FEATURES),
            artifacts={"learner": str(learner_path)},
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_reqs,
            registered_model_name=args.model_name,
        )

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{args.model_name}'")
        run_versions = [v for v in versions if v.run_id == run.info.run_id]
        if not run_versions:
            raise RuntimeError("Could not find the registered model version for this run.")

        created_version = max(run_versions, key=lambda v: int(v.version)).version

        client.transition_model_version_stage(
            name=args.model_name,
            version=created_version,
            stage=args.stage,
            archive_existing_versions=True,
        )

        print(f"MLFLOW_TRACKING_URI={tracking_uri}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Registered model: {args.model_name} (version {created_version} -> {args.stage})")


if __name__ == "__main__":
    main()
