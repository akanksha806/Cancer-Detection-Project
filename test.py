import mlflow.pyfunc

model = mlflow.pyfunc.load_model(
    model_uri="models:/CancerDetectionModel/1"
)