import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# print(np.sum(layer_outputs, axis=1))  # axis 1 keeps same dimension as input

class NeuralNetwork:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.rand(inputs, neurons)
        self.bias = np.zeros((1, neurons))

    # forward_propagation
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


# activation fns
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

        # data sets


X, y = spiral_data(samples=100, classes=3)

# input 2 neuron with 3 outputs
first_layer = NeuralNetwork(2, 3)  # create weights and biases
first_layer.forward(X)
l1_output = first_layer.output
# firstlayer activated with relu
actiavtion1 = Activations.relu(l1_output)
print(f" Relu activation weghts\n {actiavtion1[:5]}")
second_layer = NeuralNetwork(3, 3)
second_layer.forward(actiavtion1)
# firstlayer activated with softmax

activaion2 = Activations.softmax(second_layer.output)
print(f" Softmax activation weghts\n {activaion2[:5]}")
