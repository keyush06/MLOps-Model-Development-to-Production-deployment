from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


# from ml_pipeline import run_ml_pipeline
# import sys
# import os
# # sys.path.append("/opt/airflow/src")
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from data import preprocess_df
from features import create_feature_matrix
from train import train
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from mlflow.tracking import MlflowClient

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def preprocess_data(**context):
    train_path = "/opt/airflow/data/green_tripdata_2025-01.parquet"
    val_path = "/opt/airflow/data/green_tripdata_2025-02.parquet"

    df_train = preprocess_df(train_path)
    df_val = preprocess_df(val_path)

    context['ti'].xcom_push(key='X_train', value=df_train.drop(columns=['duration']))
    context['ti'].xcom_push(key='y_train', value=df_train['duration'].tolist()) ## changed to list to avoid serialization errors
    context['ti'].xcom_push(key='X_val', value=df_val.drop(columns=['duration']))
    context['ti'].xcom_push(key='y_val', value=df_val['duration'].tolist())

def train_task(**context):
    X_train = context['ti'].xcom_pull(task_ids='preprocess', key='X_train')
    y_train = pd.Series(context['ti'].xcom_pull(task_ids='preprocess', key='y_train'))
    X_val = context['ti'].xcom_pull(task_ids='preprocess', key='X_val')
    y_val = pd.Series(context['ti'].xcom_pull(task_ids='preprocess', key='y_val'))

    ## converts to DataFrame that was changed to dictionary records in the preprocess task
    X_train = pd.DataFrame(X_train)
    X_val  = pd.DataFrame(X_val)

    X_train, dv = create_feature_matrix(X_train)
    X_val, _ = create_feature_matrix(X_val, dv=dv, fit_dv=False)


    model, rmse = train(
    X_train, X_val, y_train, y_val, model_type='xgb')


with DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    description='An ML Pipeline DAG',
    # schedule_interval='@daily',
    start_date=datetime(2023, 10, 1),
    catchup=False
) as dag:
    
    preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=preprocess_data,
        # provide_context=True
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_task,
        # provide_context=True
    )

    preprocess >> train_model
