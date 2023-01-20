# Caleb Smith
# 01/19/2023
import neuralnet
class neuralnetbuilder:
    def __init__(self):
        self.bias = None
        self.learning_rate = None
        self.neural_layers = None
        self.layer_weights = None

    def set_bias(self, bias):
        self.bias = bias

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_neural_layers(self, neural_layers):
        self.neural_layers = neural_layers

    def set_layer_weights(self, layer_weights):
        self.layer_weights = layer_weights

    def build(self):
        return neuralnet.neuralnetwork(self.learning_rate, self.neural_layers)

class neurallayerbuilder:
    def __init__(self):
        self.activation_type = None
        self.input_size = None

    def set_activation_type(self, activation_type):
        self.activation_type = activation_type

    def set_input_size(self, input_size):
        self.input_size = input_size

    def build(self):
        return neuralnet.neurallayer(self.activation_type, self.input_size)
        