
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the training and testing data
df_train = pd.read_csv('pred_air_quality/Train.csv')
df_test = pd.read_csv('pred_air_quality/Test/Test.csv')

# Extract features and target labels from training data
X_train = df_train.iloc[:, :-1].values
Y_train = df_train.iloc[:, -1].values

# Extract features from testing data
X_test = df_test.values

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, Y_train)
logistic_predictions = logistic_model.predict(X_test_scaled)

# Artificial Neural Network (ANN) Model
ann_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
ann_model.fit(X_train_scaled, Y_train)
ann_predictions = ann_model.predict(X_test_scaled)

# Calculate the average of predictions
average_predictions = (logistic_predictions + ann_predictions) / 2

# Write the results to CSV
with open("result_average.csv", 'w') as f:
    f.write("Id,target\n")
    for i, pred in enumerate(average_predictions):
        f.write(f"{i},{pred}\n")
