from sklearn.metrics import mean_squared_error 

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    return rmse
