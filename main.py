from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

app = FastAPI()

# model = mlflow.pyfunc.load_model(
#     model_uri="models:/CancerDetectionModel/1"
model = mlflow.pyfunc.load_model(
    "mlruns/828349877872341255/daa33714270643abb4dd920fd47dd154/artifacts/Logistic_Regression"
)
# artificats url

class CancerData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


@app.post("/predict")
def predict(data: CancerData):

    df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(df)

    return {
        "prediction": prediction.tolist()
    }

# import mlflow.pyfunc
# model = mlflow.pyfunc.load_model(
# model_uri="models:/CancerDetectionModel/1"
# )
# import io
# import pandas as pd

# # Your raw CSV data string
# csv_data = """radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst,
# 17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189"""

# # Read the string into a DataFrame
# df = pd.read_csv(io.StringIO(csv_data))

# # Drop the empty column created by the trailing comma
# df = df.dropna(how="all", axis=1)


# prediction = model.predict(df)
# print(prediction)