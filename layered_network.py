import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()


class NeuralNetwork:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.rand(inputs, neurons)
        self.bias = np.zeros((1, neurons))

    # forward_propagation
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


class Activations:
    @staticmethod
    def relu(inputs):
        return np.maximum(0, inputs)


# data sets

X, y = spiral_data(samples=100, classes=3)

# input 2 neuron with 3 outputs
first_layer = NeuralNetwork(2, 3)  # create weights and biases
first_layer.forward(X)

activation1 = Activations.relu(first_layer.output)
# next layer with activation1 as input
second_layer = NeuralNetwork(3, 3)  # next layer with 3 in from prev layer and 3 out
second_layer.forward(activation1)  # we pass the output to the next layer
activation2 = Activations.relu(second_layer.output)

# print(f"Before activation {second_layer.output[:20]}")
# print(f"After activation using Relu {activation2[:20]}")
