from sklearn.ensemble import RandomForestRegressor
import joblib
import mlflow
import mlflow.sklearn

def train_model(X_train, y_train, model_params, save_path="C:/Users/Anushka/Documents/personal/BITS/Projects/ml-pipeline-house-price/models/model.pkl"):
    with mlflow.start_run():
        model = RandomForestRegressor(**model_params)
        model.fit(X_train,y_train)
        mlflow.sklearn.log_model(model,"model")
        joblib.dump(model, save_path)
        return model