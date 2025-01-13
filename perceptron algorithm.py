import numpy as np
def perceptron(x, y, lr=0.01, epochs=100):
    n_samples, n_features = x.shape
    weights = np.zeros(n_features)  # 1d array of size of n_features with zeroes
    bias = 0
    for epoch in range(epochs):
        for idx, i in enumerate(x):
            linear_out = np.dot(weights, i) + bias
            y_pred = np.sign(linear_out)
            if y_pred != y[idx]:
                weights += lr * y[idx] * i  # updating weights
                bias += lr * y[idx]  # updating biases
    return weights,bias

# Sample training data (AND gate)
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([-1, -1, -1, 1])  # AND gate labels (-1 for False, 1 for True)
# sample training for other gates

X =input("Enter the logic input sperated  by spaces: ")
x_array = np.array([float(i) for i in X.split()]).reshape(-1, 2)    # entered inputs are reshaped
print(x_array)
Y = input("Enter the logic outputs sperated by spaces: ")
y_array=np.array([float(i) for i in Y.split()])
print(y_array)

# Train the perceptron
weights, bias = perceptron(x_array, y_array, lr=0.1, epochs=10)
print(weights)
print(bias)

# testing the model:
def predict(x, test_weights, test_bias):
    linear_output=np.dot(test_weights,x)+test_bias
    return np.sign(linear_output)
test=input("enter test case")
test_data=np.array(([float(i) for i in test.split()]))
prediction=predict(test_data, weights, bias)
print(f"prediction of the test case {test_data} is {prediction}")