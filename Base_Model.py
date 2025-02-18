import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data

nnfs.init()


class NeuralNetwork:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.rand(inputs, neurons)
        self.bias = np.zeros((1, neurons))

    # Forward propagation
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

    def backward(self, dvalues, lr):
        3\
        #findind derivatives
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)

#new weights adn biases are updated Wnew=Wold-lr*dl/dw
        self.bias -=lr*self.dbias
        self.weights -= lr * self.dweights


# Activation functions
class Activations:
    @staticmethod
    def relu(inputs):
        return np.maximum(0, inputs)

    def relu_derivative(dvalues, inputs):
        dvalues_copy = dvalues.copy()  # Prevent in-place mutation
        dvalues_copy[inputs <= 0] = 0  # Gradient is zero where input was negative
        return dvalues_copy

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

    def backward(self, dvalues, targets):
        samples = len(dvalues)
        dvalues_copy = dvalues.copy()#not changing og values
        dvalues[range(samples), targets] -= 1
        return dvalues / samples


# Create dataset
# X, y = spiral_data(samples=100, classes=3)
X, y = vertical_data(samples=100, classes=3)

# First layer: 2 inputs, 3 outputs
# Initialize layers
first_layer = NeuralNetwork(2, 3)
second_layer = NeuralNetwork(3, 3)
loss_function = Cross_EntropyLoss()
learning_rate = 0.1

# Apply ReLU activation
# Training loop
for epoch in range(10000):
    # Forward pass
    first_layer.forward(X)
    activation1 = Activations.relu(first_layer.output)
    second_layer.forward(activation1)
    activation2 = Activations.softmax(second_layer.output)

    # Loss calculation
    loss = loss_function.calculate(activation2, y)
    predictions = np.argmax(activation2, axis=1)  # finding highes indexed number=>corresppodning class
    accuracy = np.mean(predictions == y)  # chechk with origianl values adn returns a boolen array

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")

    # Backpropagation
    dloss = loss_function.backward(activation2, y)
    second_layer.backward(dloss, learning_rate)
    dactivation1 = Activations.relu_derivative(np.dot(dloss, second_layer.weights.T), first_layer.output)
    first_layer.backward(dactivation1, learning_rate)

print("Training Complete!")
