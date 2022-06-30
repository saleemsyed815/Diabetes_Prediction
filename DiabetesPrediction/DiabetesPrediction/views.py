from django.shortcuts import render

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    data = pd.read_csv(r'C:/Users/Admin/Desktop/diabetes/kaggle_diabetes.csv')

    dataset_new = dataset

    #replacing 0 values with Nan
    dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)

    # Replacing NaN with mean values
    dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
    dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
    dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
    dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
    dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)

    # Feature scaling using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    dataset_scaled = sc.fit_transform(dataset_new)

    dataset_scaled = pd.DataFrame(dataset_scaled)

    # Selecting features - [Glucose, Insulin, BMI, Age]
    X = dataset_scaled.iloc[:, [1, 4, 5, 7]].values
    Y = dataset_scaled.iloc[:, 8].values

    # Splitting X and Y
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )

    # Random forest Algorithm
    from sklearn.ensemble import RandomForestClassifier
    ranforest = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
    ranforest.fit(X_train, Y_train)

    # prediction = ranforest.predict(X_test)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])

    prediction = ranforest.predict([[val1, val2, val3, val4]])

    result1 = ""
    if prediction==[1]:
        result1 = "Positive"
    else:
        result1 = "Negative"


    return render(request, 'predict.html', {"result2":result1})
