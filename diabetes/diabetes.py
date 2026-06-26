import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv('diabetes.csv')

zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_columns] = df[zero_columns].replace(0, np.nan)

for col in zero_columns:
    df[col] = df[col].fillna(df[col].median())
print(df.describe())
print(df.info())
target = 'Outcome'
x = df.drop(columns=[target])
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print(classification_report(y_test, y_predict))

with open("diabetes.pkl", "wb") as file:
    pickle.dump(model, file)