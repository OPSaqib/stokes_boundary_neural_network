import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Linear Regression

def msre(input, target):
    return np.mean(((input - target) / target) ** 2)

data = pd.read_csv('kaggle_train_Stokes.csv')

X = np.column_stack([
    (data['h']),
    (data['nu'] / (data['omega'] * data['h'])),
    ((data['omega'] * (data['h'] ** 3)) / data['nu'])
])

y = data['z*']

# Run model
model = LinearRegression()
model.fit(X, y)

w0 = model.intercept_
w1, w2, w3 = model.coef_

r2 = model.score(X, y)

print(f"Coefficents: w0 = {w0}, w1 = {w1}, w2 = {w2}, w3 = {w3}")
print(f"R^2 value: {r2}")

# See how good regression model is by seeing it's MSRE
y_pred = model.predict(X)

total_msre = 0
for i in range(len(X)):
    msre_i = msre(y_pred[i], y[i])
    total_msre += msre_i

avg_msre = total_msre / len(X)

print(f"Average MSRE: {avg_msre}")
