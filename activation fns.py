from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

class ActivationFns:
    def __init__(self):
        pass

    def linear_out(self, inputs, weight, bias):
        # Linear out y = mx + c
        output = np.dot(inputs, weight) + bias
        return output

    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def relu(self, inputs):
        return np.maximum(0, inputs)

    def tanh(self, inputs):
        return np.tanh(inputs)

# Generate dataset
X, y = make_regression(
    n_samples=100,  # Number of samples
    n_features=1,   # Number of features
    noise=10,       # Noise level (adds randomness)
    random_state=42,  # Seed for reproducibility
)

n_features = X.shape[1]
weights = np.random.randn(n_features)  # Random weights
bias = np.random.randn(1)  # Random bias

# Make an instance of class
linear = ActivationFns()

# Linear output
output_fn = linear.linear_out(X, weights, bias)
print(f"original output before activation: {output_fn[:5]}")

# Apply activation functions
sigmoid_output = linear.sigmoid(output_fn)
relu_output = linear.relu(output_fn)
tanh_output = linear.tanh(output_fn)

# Printing activation function outputs (first 5 for clarity)
print(f"sigmoid_output: {sigmoid_output[:5]}")
print(f"relu_output: {relu_output[:5]}")
print(f"tanh_output: {tanh_output[:5]}")

# Plotting
plt.scatter(X, y, color='blue', alpha=0.7, label='Original Data')
plt.plot(X, sigmoid_output, color='red', label='sigmoid')
plt.plot(X, relu_output, color='green', label='relu')
plt.plot(X, tanh_output, color='yellow', label='tanh')
plt.title("Generated Linear Regression Dataset")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()
