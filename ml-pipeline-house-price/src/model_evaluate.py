from sklearn.metrics import mean_squared_error, r2_score
import mlflow

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    mlflow.log_metric("rmse",rmse)
    mlflow.log_metric("r2_score",r2)
    
    print(f"RMSE: {rmse:.3f}, R2: {r2:.3f}")
    return rmse, r2