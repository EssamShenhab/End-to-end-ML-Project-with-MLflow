import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import dagshub
from ML_Project.entity.config_entity import ModelEvaluationConfig
from ML_Project.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize with a config object containing evaluation settings.
        """
        self.config = config

    def eval_metrics(self, actual, pred):
        """
        Compute evaluation metrics.
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        """
        Evaluate model, log metrics and model to MLflow, and save metrics locally.
        """
        try:
            dagshub.init(repo_owner='EssamShenhab', repo_name='End-to-end-ML-Project-with-MLflow', mlflow=True)
            # Load test data and model
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)

            test_x = test_data.drop(columns=[self.config.target_column])
            test_y = test_data[self.config.target_column].values  # Convert to 1D array

            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted_qualities = model.predict(test_x)

                # Evaluate
                rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

                # Save metrics locally
                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                save_json(path=Path(self.config.metric_file_name), data=scores)

                # Log parameters and metrics
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

                # Log model (and register if not file-based)
                input_example = test_x.iloc[:1]
                signature = infer_signature(test_x, predicted_qualities)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        registered_model_name="ElasticnetModel",
                        input_example=input_example,
                        signature=signature
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        input_example=input_example,
                        signature=signature
                    )

        except Exception as e:
            raise RuntimeError(f"Model Evaluation failed: {e}")
