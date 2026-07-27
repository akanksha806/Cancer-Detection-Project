from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import sys
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

x_train = preprocessor.preprocess_data(x_train)
x_test = preprocessor.preprocess_data(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)