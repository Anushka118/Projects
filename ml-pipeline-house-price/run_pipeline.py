import yaml
from src.data_preprocessing import load_data, preprocess_data
from src.model_train import train_model
from src.model_evaluate import evaluate_model
import mlflow

if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    mlflow.set_experiment("My_Machine_Learning_Project")
    with mlflow.start_run() as run:
        # 1. Log Preprocessing Parameters
        mlflow.log_param("target_column", config["target_column"])
        mlflow.log_param("test_size", config["test_size"])
        mlflow.log_param("random_state", config["random_state"])
        
        # 2. Log Model Parameters
        # It's better to log the individual model parameters from the 'params' dictionary
        if "model" in config and "params" in config["model"]:
            mlflow.log_params(config["model"]["params"])
    
        df = load_data(config["data_path"])
        X_train, X_test, y_train, y_test = preprocess_data(
                df,
                target_col = config["target_column"],
                test_size = config["test_size"],
                random_state = config["random_state"]
            )
    
        model = train_model(X_train, y_train, config["model"]["params"])
        evaluate_model(model, X_test, y_test)
