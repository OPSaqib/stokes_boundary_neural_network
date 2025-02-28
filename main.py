import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Initializing weights function for 1 hidden layer neural network

def initialize_weights(input_size, num_neurons, output_size):
    W1 = np.random.normal(0, 0.2, (num_neurons, input_size))
    W3 = np.random.normal(0, 0.2, (output_size, num_neurons))
    return W1, W3

# Relu function

def relu(x):
    return np.maximum(0, x)

# Heaviside function

def heaviside(x):
    return np.where(x >= 0, 1, 0)

# msre function

def msre(input, target):
    
    input_new = np.exp(input)
    target_new = np.exp(target)
    return np.mean(((input_new - target_new) / target_new) ** 2)

# Forward pass function
def forward_pass(x, W1, W3):
    X1 = W1 @ x + 0.1
    X2 = relu(X1)
    X3 = W3 @ X2
    return X1, X2, X3

# Backward pass function
def backward_pass(x, y, X1, X2, X3, W1, W3, lr):
    X3_new = np.exp(X3)
    y_new = np.exp(y)

    # Derivatives
    dE_dX3 = 2 * (X3_new - y_new) / (y_new ** 2)
    dE_dX2 = W3.T @ dE_dX3
    dE_dX1 = dE_dX2 * heaviside(X1)

    dE_dW3 = dE_dX3 @ X2.T
    dE_dW1 = dE_dX1 @ x.T
    
    W1 -= lr * dE_dW1
    W3 -= lr * dE_dW3

    return W1, W3

# Predict from csv

def predict_from_csv(W1, W3, x_mean, x_std, y_mean, y_std):
    test_data = pd.read_csv("kaggle_test_Stokes.csv")

    x = np.column_stack([
        np.log(test_data['h']),
        np.log(test_data['nu'] / (test_data['omega'] * test_data['h'])),
        np.log((test_data['omega'] * (test_data['h'] ** 3)) / test_data['nu'])
    ])

    x = (x - x_mean) / x_std
    ids = test_data['id'].values
    predictions = []
    
    for i in range(len(x)):
        X1, X2, Z2 = forward_pass(x[i].reshape(-1, 1), W1, W3)
        prediction = np.exp((Z2.item() * y_std) + y_mean)
        predictions.append(prediction)
    
    output_df = pd.DataFrame({'id': ids, 'prediction': predictions})
    output_df.to_csv("kaggle_predictions_group_8.csv", index=False)

# Train the model

def train_model(x_data, y_data, input_size, num_neurons, output_size, y_mean, y_std, epochs=600, lr=0.001):

    # Init weights
    W1, W3 = initialize_weights(input_size, num_neurons, output_size)
    
    for epoch in range(epochs):
        sum_loss = 0
        indices = np.random.permutation(len(x_data)) #For SGD mix up the data
        for i in indices:
            x = x_data[i].reshape(-1, 1)
            y = y_data[i]
            X1, X2, X3 = forward_pass(x, W1, W3)
            loss = msre(X3, y)
            sum_loss += loss
            W1, W3 = backward_pass(x, y, X1, X2, X3, W1, W3, lr)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {sum_loss}")
            print(f"W1:\n{W1}\nW3:\n{W3}")
    return W1, W3

# Main method

def main():
    data = pd.read_csv('kaggle_train_Stokes.csv')
    x = np.column_stack([
        np.log(data['h']),
        np.log(data['nu'] / (data['omega'] * data['h'])),
        np.log((data['omega'] * (data['h'] ** 3)) / data['nu'])
    ])

    # Standardize x data (inputs)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x = (x - x_mean) / x_std
    
    # Standardize y data (outputs)
    y = np.log(data['z*'].values)
    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean) / y_std
    
    input_size = 3
    num_neurons = 100
    output_size = 1

    W1, W3 = train_model(x, y_normalized, input_size, num_neurons, output_size, y_mean, y_std)

    # Predict from test_Stokes
    predict_from_csv(W1, W3, x_mean, x_std, y_mean, y_std)

    return W1, W3

# Method to check overfitting

def overfitting(W1, W3):

    data = pd.read_csv('kaggle_train_Stokes.csv')
    x = np.column_stack([
        np.log(data['h']),
        np.log(data['nu'] / (data['omega'] * data['h'])),
        np.log((data['omega'] * (data['h'] ** 3)) / data['nu'])
    ])

    # Standardize x data (inputs)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x = (x - x_mean) / x_std
    
    # Standardize y data (outputs)
    y = np.log(data['z*'].values)
    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean) / y_std
    
    # Split
    split = int(0.8 * len(x))  

    x_80 = x[:split] 
    x_20 = x[split:]  
    y_80 = y_normalized[:split]  
    y_20 = y_normalized[split:]
    
    # Run model get W1, W3
    input_size = 3
    num_neurons = 100
    output_size = 1

    # Compute average MSRE for 80,20 compare them
    total_msre_80 = 0
    for i in range(len(x_80)):
        X1, X2, y_pred_80 = forward_pass(x_80[i], W1, W3)
        msre_80 = msre(y_pred_80, y_80[i])
        total_msre_80 += msre_80

    total_msre_20 = 0
    for i in range(len(x_20)):
        X1, X2, y_pred_20 = forward_pass(x_20[i], W1, W3)
        msre_20 = msre(y_pred_20, y_20[i])
        total_msre_20 += msre_20
    
    avg_msre_80 = total_msre_80 / len(x_80)
    avg_msre_20 = total_msre_20 / len(x_20)

    #avg_msre_80 should be approx equal to avg_msre_20 if no overfitting or any underfitting

    print(f"Average MSRE (80% Training Data): {avg_msre_80:.5f}")
    print(f"Average MSRE (20% Test Data): {avg_msre_20:.5f}")

# Run everything

if __name__ == "__main__":
    W1, W3 = main()
    overfitting(W1, W3)
