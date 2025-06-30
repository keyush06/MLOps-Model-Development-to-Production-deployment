from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from src import hyperparam_optimization
from hyperparam_optimization import run_hyperopt

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import mlflow
from sklearn.metrics import mean_squared_error


""""Train a model based on the provided training data and validation data.
Args:
    x_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target variable.
    model_type (str): Type of model to train ('linear', 'xgb').
    linear_type (str, optional): Type of linear model ('ridge', 'lasso'). Defaults to None.
    alpha (float, optional): Regularization strength for ridge or lasso. Defaults to 0.1.
    
    
    I am trying to make the code more modular but otherwise we could create different functions for linear regression and Boosting.
    """



def train(X_train, X_val, y_train, y_val, model_type = "linear", linear_type = None, alpha=0.1):
    best_params = run_hyperopt(X_train, y_train, X_val, y_val, model_type=model_type, linear_type=linear_type)

    if model_type == "xgb":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        y_pred = model.predict(dval)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(model, artifact_path="models_mlflow_v2")

    elif model_type == "linear":
        if linear_type == "ridge":
            model = Ridge(alpha=best_params['alpha'])
        elif linear_type == "lasso":
            model = Lasso(alpha=best_params['alpha'])
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="models_mlflow_v2")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Model trained with {model_type} and validation RMSE: {rmse}")
    return model, rmse