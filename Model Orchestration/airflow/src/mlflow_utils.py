import mlflow
from src import evaluate

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment_workflow")

def start_run_and_log(model, dv, X_val, y_val, alpha, train_path, val_path):
    with mlflow.start_run():
        mlflow.set_tag("developer", "keyush")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("train-dataPath", train_path)
        mlflow.log_param("val-dataPath", val_path)
        rmse = evaluate.evaluate_model(model, X_val, y_val)
        mlflow.log_metric("rmse", rmse)
