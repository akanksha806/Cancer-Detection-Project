from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import sys
import mlflow
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

mlflow.set_experiment("Cancer Detection Model Experiment")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datapreprocessing.preprocess import Data

PATH = r"C:\Users\Administrator\CancerDetectionProject\data\Cancer_Data.csv"

data = pd.read_csv(PATH)

x = data.drop(['diagnosis', 'Unnamed: 32', 'id'], axis=1)
y = data['diagnosis']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

preprocessor = Data()

transformer = preprocessor.preprocess_data(x_train)

x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)


#from here the model experiment is starting
with mlflow.start_run():
    model = LogisticRegression()
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

#here we are calculate the metrices of the model
    acc_score = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,pos_label='M')
    recall = recall_score(y_test,y_pred,pos_label='M')
    score = f1_score(y_test,y_pred,pos_label='M')

    # storing the model metrices
    mlflow.log_metric("accuracy",acc_score)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("F1_score",score)

    #logging the model
    mlflow.sklearn.log_model(model, artifact_path="LogisticRegression")

    # message
    print("model saved succesfully")