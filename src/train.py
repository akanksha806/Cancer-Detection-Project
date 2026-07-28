from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

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
    x, y,
    test_size=0.2,
    random_state=42
)

preprocessor = Data()

transformer = preprocessor.preprocess_data(x_train)

x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
   "Naive Bayes": GaussianNB(),
    # "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

for name, algo in models.items():

    # MLflow Run
    with mlflow.start_run(run_name=name):

        # Train Model
        algo.fit(x_train, y_train)

        # Prediction
        y_pred = algo.predict(x_test)

        # Metrics
        acc_score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='M')
        recall = recall_score(y_test, y_pred, pos_label='M')
        score = f1_score(y_test, y_pred, pos_label='M')
        cm = confusion_matrix(y_test, y_pred)

        # Log Metrics
        mlflow.log_metric("accuracy", acc_score)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", score)

        # Log Model
        mlflow.sklearn.log_model(
            sk_model=algo,
            artifact_path=name.replace(" ", "_")
        )

        # Print Results
        print(f"\n{name}")
        print(f"Accuracy : {acc_score:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {score:.4f}")
        print("Confusion Matrix:")
        print(cm)

        print(f"{name} has been saved successfully!")