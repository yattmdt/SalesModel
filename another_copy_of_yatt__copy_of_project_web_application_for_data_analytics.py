# -*- coding: utf-8 -*-
"""Another copy of Yatt__Copy of Project - Web Application for Data Analytics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G5-uM2O1JkJXoMnrOraAi5lp4e6HK0Br

# Project - Web Application for Data Analytics

## About the instructor:

**Ts. Nur Shakirah Md Salleh**

Lead Technical Trainer - Data, Analytic and Machine Learning

airasia academy

nurshakirahmdsalleh@airasiaacademy.com

LinkedIn: [Ts. Nur Shakirah Md Salleh](https://www.linkedin.com/in/nurshakirahmdsalleh)

©2023 [airasia academy](https://airasiaacademy.com) All rights reserved.

#Project

Instructions:
1. Develop a Python website to predict the target and deploy it using Streamlit using GitHub repository.
2. You are require to use `Advertising.csv` dataset (Team Krypton) and `iris` dataset (Team Asgard) in this case study. Generate the model of this dataset outside of the streamlit environment (You can use Google Colab). You just need to load the model in this app (no model training should happens in this application).
3. You are require to choose only **ONE** of the most suitable supervised machine learning algorithm to solve this problem.
4. Feature scaling is optional for this project.
5. You need to enable slider to set the feature values.


**Submission**

Submit the following information:
* source code to generate the model (.pdf)
* screenshot of the source code (.py) on GitHub
* screenshot of the streamlit app
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/Advertising - Advertising.csv')

df.head()

df.tail()

df.info()

df.describe()

df.dtypes

df.duplicated().sum()

df.describe()

df.head()

df.drop('Unnamed: 0', axis=1, inplace=True)

df.head()

df.hist()

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled)
df_scaled.columns = df.columns
df_scaled.head()

X = df_scaled.drop('Sales',axis=1)
y = df_scaled.Sales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=100)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

from sklearn.svm import SVR

modelSvr = SVR(kernel='linear', C=3).fit(X_train, y_train)
y_pred = modelSvr.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("MAE:{}".format(mean_absolute_error(y_test,y_pred)))
print("MSE:{}".format(mean_squared_error(y_test,y_pred)))
print("RMSE:{}".format(mean_squared_error(y_test,y_pred)**0.5))
print("R2:{}".format(r2_score(y_test,y_pred)**0.5))

y_pred

df_actual = pd.DataFrame()
df_actual = X_test.copy()
df_actual['Actual'] = y_test
df_actual

df_prediction = pd.DataFrame()
df_prediction = X_test.copy()
df_prediction['Predicted'] = y_pred
df_prediction

unscaleddf_actual = scaler.inverse_transform(df_actual)
unscaleddf_actual = pd.DataFrame(unscaleddf_actual)
unscaleddf_actual.columns = df_actual.columns

unscaleddf_prediction = scaler.inverse_transform(df_prediction)
unscaleddf_prediction = pd.DataFrame(unscaleddf_prediction)
unscaleddf_prediction.columns = df_prediction.columns

unscaleddf_actual.head()

unscaleddf_prediction.head()

df_actualpredicted = pd.DataFrame()
df_actualpredicted['Actual'] = unscaleddf_actual['Actual'].copy()
df_actualpredicted['Predicted'] = unscaleddf_prediction['Predicted'].copy()
df_actualpredicted.head()

import matplotlib.pyplot as plt

df_actualpredicted.plot(kind="bar", figsize=(50,20))

plt.title('Sales Target')
plt.xlabel('Index')
plt.ylabel('Sales')

df_actualpredicted.to_csv("Sales-Target.csv")

import pickle
pickle.dump(modelSvr, open ('Sales-Target.h5','wb'))