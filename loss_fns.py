import numpy as np
import nnfs
from nnfs.datasets import spiral_data,vertical_data

nnfs.init()


class NeuralNetwork:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.rand(inputs, neurons)
        self.bias = np.zeros((1, neurons))

    # Forward propagation
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


# Activation functions
class Activations:
    @staticmethod
    def relu(inputs):
        return np.maximum(0, inputs)

    @staticmethod
    def softmax(inputs):
        # Normalizing the values to avoid overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Calculating the probabilities
        prob_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return prob_values


# Base Loss class
class Loss:
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss


class Cross_EntropyLoss(Loss):
    def forward(self, inputs, targets):
        samples = len(inputs)

        # Clip data to prevent division by zero
        clipped_inputs = np.clip(inputs, 1e-7, 1 - 1e-7)

        # Check the shape of the targets
        if len(targets.shape) == 1:  # Categorical labels
            correct_confidences = clipped_inputs[range(samples), targets]
        elif len(targets.shape) == 2:  # One-hot encoded labels
            correct_confidences = np.sum(clipped_inputs * targets, axis=1)
        else:
            raise ValueError("Invalid shape for targets. Must be 1D or 2D.")

        # Calculate negative log likelihoods
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# Create dataset
X, y = spiral_data(samples=100, classes=3)
# X, y = vertical_data(samples=100, classes=3)

# First layer: 2 inputs, 3 outputs
first_layer = NeuralNetwork(2, 3)
first_layer.forward(X)
l1_output = first_layer.output

# Apply ReLU activation
activation1 = Activations.relu(l1_output)
print(f"Relu activation outputs:\n{activation1[:5]}")

# Second layer: 3 inputs, 3 outputs
second_layer = NeuralNetwork(3, 3)
second_layer.forward(activation1)

# Apply Softmax activation
activation2 = Activations.softmax(second_layer.output)
print(f"Softmax activation outputs:\n{activation2[:5]}")

# Calculate Cross-Entropy Loss
loss_function = Cross_EntropyLoss()
loss = loss_function.calculate(activation2, y)

print(f"Loss: {loss}")


lowest_loss = np.inf  # Initialize the lowest loss to infinity

# Acquiring best weights and biases
best_layer1_weights = first_layer.weights.copy()
best_layer2_weights = second_layer.weights.copy()
best_layer1_bias = first_layer.bias.copy()
best_layer2_bias = second_layer.bias.copy()

# Updating weights and checking losses
for iteration in range(10000):
    # Randomly initialize weights and biases
    first_layer.weights = 0.05 * np.random.rand(2, 3)
    second_layer.weights = 0.05 * np.random.rand(3, 3)
    first_layer.bias = 0.05 * np.random.rand(1, 3)
    second_layer.bias = 0.05 * np.random.rand(1, 3)

    # Forward pass through the first layer and ReLU activation
    first_layer.forward(X)
    activation1 = Activations.relu(first_layer.output)

    # Forward pass through the second layer and softmax activation
    second_layer.forward(activation1)
    activation2 = Activations.softmax(second_layer.output)

    # Calculate the loss
    loss = loss_function.calculate(activation2, y)

    # Accuracy calculation
    predictions = np.argmax(activation2, axis=1)
    accuracy = np.mean(predictions == y)

    # Print accuracy for monitoring
    # print(f"Iteration {iteration}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")

    # Update best weights and biases if the loss is the lowest
    if loss < lowest_loss:
        print(f"New set of weights found ,iterations ,{iteration}, loss ,{loss}, acc ,{accuracy}")
        lowest_loss = loss
        best_layer1_weights = first_layer.weights.copy()
        best_layer2_weights = second_layer.weights.copy()
        best_layer1_bias = first_layer.bias.copy()
        best_layer2_bias = second_layer.bias.copy()

    else:
        first_layer.weights = best_layer1_weights.copy()
        second_layer.weights = best_layer2_weights.copy()
        first_layer.bias = best_layer1_bias.copy()
        second_layer.bias = best_layer2_bias.copy()