# Toy MLOps Pipeline

Local-first toy MLOps system that will include:
- Training with fastai (Iris) + MLflow tracking and registry
- FastAPI inference service loading model from MLflow
- Docker + kind Kubernetes deployment
- Prometheus + Grafana monitoring
- Kubeflow Pipelines v2 orchestration
- Optional AWS Terraform path

## Bootstrap
```bash
make setup
make lint
make test
make serve-local

## Training (M1) commands

```powershell
make setup
make mlflow-up
# open http://localhost:5000
make train
make predict-sample
make mlflow-down
