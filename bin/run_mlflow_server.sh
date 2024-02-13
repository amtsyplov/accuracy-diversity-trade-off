source $(pwd)/.venv/bin/activate

export MLFLOW_STORAGE=$(pwd)/.mlruns/
echo $MLFLOW_STORAGE is used to storage experiments
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri $MLFLOW_STORAGE --no-serve-artifacts