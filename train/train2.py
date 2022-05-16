from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import os
from os.path import exists
from os import makedirs
from os import environ

dataset = pd.read_csv("C:/Users/bmithran/Desktop/hack2build/sensor.csv")
df = dataset.drop('sensor_15',axis = 1)
df = df.drop('Unnamed: 0',axis = 1)
df = df.drop('timestamp',axis = 1)
print("drop columns done")
for column in df.columns:
    if df[column].dtypes != 'object':
        df[column] = df[column].fillna(df[column].mean())

print("replacing null values  done")

df['machine_status'].replace(['BROKEN', 'NORMAL' , 'RECOVERING'],
                        [2 , 0, 1 ], inplace=True)
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
print("X , Y split done")

print("import imblearn done")
import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)
print("resampling done")
X_train2 , X_test2 , Y_train2 , Y_test2 = train_test_split(X , Y, test_size= 0.25, random_state=0)
print("train test split done")

classifier2= RandomForestClassifier(n_estimators= 10, criterion="entropy")
classifier2.fit(X_train2, Y_train2)
Y_pred2= classifier2.predict(X_test2)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test2, Y_pred2)
print(cm)
output_path = "C:/Users/bmithran/Desktop/hack2build/output"
model_location = os.path.join(output_path, 'predict.pkl')
joblib.dump(classifier2, model_location)
print("completed and stored as pickle file")