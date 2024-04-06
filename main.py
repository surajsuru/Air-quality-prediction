import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Multivariate Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, Y_train)
linear_predictions = linear_model.predict(X_test_scaled)

# Artificial Neural Network (ANN) Model
ann_model = Sequential([
    Dense(100, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(1)
])
ann_model.compile(optimizer='adam', loss='mean_squared_error')
ann_model.fit(X_train_scaled, Y_train, epochs=50, batch_size=32, verbose=0)
ann_predictions = ann_model.predict(X_test_scaled).flatten()

# Calculate the average of predictions
average_predictions = (linear_predictions + ann_predictions) / 2

# Write the results to CSV
with open("result_average.csv", 'w') as f:
    f.write("Id,target\n")
    for i, pred in enumerate(average_predictions):
        f.write(f"{i},{pred}\n")
