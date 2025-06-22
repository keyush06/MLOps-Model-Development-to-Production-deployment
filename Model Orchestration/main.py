"""This file is the entry point for the Model Orchestration application. It is kind of testing all the components locally."""

from src import mlflow_utils, data
from src import hyperparam_optimization
from src import train
from src import evaluate, train, features
import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient
import xgboost as xgb
import mlflow.xgboost

train_path = "/workspaces/MLOps-Model-Development-to-Production-deployment/Data/green_tripdata_2025-01.parquet"
val_path = "/workspaces/MLOps-Model-Development-to-Production-deployment/Data/green_tripdata_2025-02.parquet"

df_train = data.preprocess_df(train_path)
df_val = data.preprocess_df(val_path)

print(f"df_train size is {df_train.size}, df_val size is {df_val.size}")
print("Done till here of data imports and important libraries")

## Features extraction and Data Preprocessing

X_train, dict_vectorizer = features.create_feature_matrix(df_train)
X_val, _ = features.create_feature_matrix(df_val, dv=dict_vectorizer, fit_dv=False)

print("Done till here of feature extraction")

y_train = df_train['duration'].values
y_val = df_val['duration'].values
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")

## I am already running the best_params in the train function, so I am commenting this out for now. If you want to see the hyperparameter optimization results, you can uncomment this part.

# best_params = hyperparam_optimization.run_hyperopt(
#     X_train, y_train, X_val, y_val, model_type='xgb'
# )

# print(f"Best parameters found: {best_params}")

model, rmse = train.train(
    X_train, X_val, y_train, y_val, model_type='xgb')

print(f"Model trained with RMSE: {rmse} which is the best RMSE so far amongst all the models evaluated")