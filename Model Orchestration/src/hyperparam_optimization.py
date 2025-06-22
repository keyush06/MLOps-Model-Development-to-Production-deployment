from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from hyperparam_optimization import run_hyperopt

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import mlflow
from sklearn.metrics import mean_squared_error


# def objective_eval(params, train, valid, y_val):
#     with mlflow.start_run():
#         mlflow.set_tag("model", "xgboost_hyperopt")
#         mlflow.log_params(params)

#         booster = xgb.train(
#             params, train,
#             num_boost_round=1000,
#             evals=[(valid, 'validation')],
#             early_stopping_rounds=50,
#             verbose_eval=False
#         )
#         y_pred = booster.predict(valid)
#         rmse = mean_squared_error(y_val, y_pred, squared=False)
#         mlflow.log_metric("rmse", rmse)
#         mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
#         return {"loss": rmse, "status": STATUS_OK}

def run_hyperopt(X_train, y_train, X_val, y_val, model_type = 'linear', linear_type = None):

    def objective_eval(params):
        with mlflow.start_run():
            mlflow.set_tag("model_orchestration", f"{model_type}_hyperopt")
            mlflow.log_params(params)

            if model_type == "xgb":
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)

                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=[(dval, 'validation')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
                y_pred = booster.predict(dval)
                rmse = mean_squared_error(y_val, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
                mlflow.xgboost.log_model(booster, artifact_path="models_mlflow_v2")

            elif model_type == "linear":
                if linear_type == "ridge":
                    model = Ridge(alpha=params['alpha'])
                elif linear_type == "lasso":
                    model = Lasso(alpha=params['alpha'])

                else:
                    model = LinearRegression()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = mean_squared_error(y_val, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(model, artifact_path="models_mlflow_v2")

            return {"loss": rmse, "status": STATUS_OK}

    if model_type == "xgb":
        space = {
            'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
            'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
            'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
            'objective': 'reg:squarederror',
            'seed': 42
        }
    elif model_type == 'linear':
        space = {
            'alpha': hp.loguniform('alpha', -4, 0),  # 0.018 to 1
        }
    else:
        raise NotImplementedError()

    best_params = fmin(
        fn=objective_eval,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )

    return best_params