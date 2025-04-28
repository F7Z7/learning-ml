import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data
from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt

nnfs.init()


# Neural Network Layer
class NeuralNetwork:
    def __init__(self, inputs, neurons):
        self.weights = np.random.randn(inputs, neurons) * np.sqrt(2 / inputs)  # He Initialization
        self.bias = np.zeros((1, neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

    def backward(self, dvalues, lr):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)

        self.weights -= lr * self.dweights
        self.bias -= lr * self.dbias


# Activation Functions
class Activations:
    @staticmethod
    def relu(inputs):
        return np.maximum(0, inputs)

    @staticmethod
    def relu_derivative(der_values, inputs):
        dvalues_copy = der_values.copy()
        dvalues_copy[inputs <= 0] = 0  # Zero gradient for negative inputs
        return dvalues_copy

    @staticmethod
    def softmax(inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)


# Cross-Entropy Loss
class CrossEntropyLoss:
    def forward(self, inputs, targets):
        samples = len(inputs)
        clipped_inputs = np.clip(inputs, 1e-7, 1 - 1e-7)
        correct_confidences = clipped_inputs[range(samples), targets]
        return -np.log(correct_confidences)

    def backward(self, dvalues, targets):
        samples = len(dvalues)
        dvalues_copy = dvalues.copy()
        dvalues_copy[range(samples), targets] -= 1
        return dvalues_copy / samples


# Create dataset
# X, y =spiral_data(samples=100, classes=3)
X, y = make_moons(n_samples=200, noise=0.1)

# Visualizing Data Before Training
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.title("Data Before Training")

# Initialize Layers
layer1 = NeuralNetwork(2, 8)
layer2 = NeuralNetwork(8, 8)
layer3 = NeuralNetwork(8, 3)  # Output layer (3 classes)

loss_function = CrossEntropyLoss()
learning_rate = 0.01
previous_accuracy = 0
# Training Loop
for epoch in range(10000):
    # Forward Pass
    layer1.forward(X)
    activation1 = Activations.relu(layer1.output)

    layer2.forward(activation1)
    activation2 = Activations.relu(layer2.output)

    layer3.forward(activation2)
    activation3 = Activations.softmax(layer3.output)

    # Loss Calculation
    loss = np.mean(loss_function.forward(activation3, y))
    predictions = np.argmax(activation3, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        if accuracy < previous_accuracy:
            print("Accuracy dropped! Stopping training.")
            break

            # Update previous accuracy
        previous_accuracy = accuracy

    # Backpropagation
    d_loss = loss_function.backward(activation3, y)
    layer3.backward(d_loss, learning_rate)

    d_activation2 = Activations.relu_derivative(np.dot(d_loss, layer3.weights.T), activation2)
    layer2.backward(d_activation2, learning_rate)

    d_activation1 = Activations.relu_derivative(np.dot(d_activation2, layer2.weights.T), activation1)
    layer1.backward(d_activation1, learning_rate)

# Visualizing Predictions After Training
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap="brg")
plt.title("Data After Training (Predictions)")
plt.show()

print("Training Complete!")
