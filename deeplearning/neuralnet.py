# Caleb Smith
# 02/03/2023
import numpy as np
from enum import Enum

class LayerType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class ActivationFunction(Enum):
    NONE = 0
    SIGMOID = 1
    TANH = 2

class neuron:
    # each neuron has a value, input and output edges, and a bias
    def __init__(self):
        self.value = None
        self.value_activated = None
        self.value_activated_deriv = None
        self.node_deriv = None
        self.in_edges = []
        self.out_edges = []
        self.bias = np.random.randn()
        self.bias_error = None
        
    def delete_neuron(self):
        for i in range(self.in_edges):
            self.in_edges[i].delete_edge()
        self.in_edges.clear()
        for i in range(self.out_edges):
            self.out_edges[i].delete_edge()
        self.out_edges.clear()

    def update_value(self, value):
        self.value = value

class edge:
    # each edge defines a weighted relationship between an input and output node
    def __init__(self, input_node, output_node):
        self.input_node = input_node
        self.output_node = output_node
        self.weight = np.random.randn()
        self.weight_error = None

    def delete_edge(self):
        self.input_node.out_edges.remove(self)
        if (len(self.input_node.out_edges) == 0):
            self.input_node.delete_neuron
        self.output_node.in_edges.remove(self)
        if (len(self.output_node.in_edges) == 0):
            self.output_node.delete_neuron
        self.input_node = None
        self.output_node = None
        
    def set_node_edges(self):
        self.input_node.out_edges.append(self)
        self.output_node.in_edges.append(self)

    def update_weight(self, weight):
        self.weight = weight

class neural_layer:
    # each neural layer has an array of neurons, a layer type, and an activation type and is responsible for activating its nodes
    def __init__(self, activation_type, neurons, layer_type=LayerType.HIDDEN):
        self.activation_type = activation_type
        self.neurons = neurons
        self.layer_type = layer_type

    def sigmoid(self, x):
        # 1/(1 + np.exp(-x)) or 1/(1 + e^-x)
        x = 1 / (1 + np.exp(-x))
        return x

    def sigmoid_deriv(self, x):
        # (1/(1 + np.exp(-x))) * (1 - (1/(1 + np.exp(-x)))) or (1/(1 + e^-x)) * (1 - (1/(1 + e^-x)))
        x = self.sigmoid(x) * (1 - self.sigmoid(x))
        return x

    def tanh(self, x):
        # (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) or (e^x – e^-x) / (e^x + e^-x)
        x = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        return x

    def tanh_deriv(self, x):
        # 1 - (_tanh(x) ** 2) or 1 - tanh^2(x)
        x = 1 - (self.tanh(x) ** 2)
        return x

    def layer_activation(self):
        if self.activation_type == 0:
            return
        if self.activation_type == 1:
            for i in range(len(self.neurons)):
                self.neurons[i].value_activated = self.sigmoid(self.neurons[i].value)
            return
        if self.activation_type == 2:
            for i in range(len(self.neurons)):
                self.neurons[i].value_activated = self.tanh(self.neurons[i].value)
            return

    def layer_deriv(self):
        layer_derivative = neural_layer(self.activation_type, self.neurons, self.layer_type)
        if self.activation_type == 0:
            return
        if self.activation_type == 1:
            for i in range(len(self.neurons)):
                layer_derivative.neurons[i].value_activated_deriv = self.sigmoid_deriv(self.neurons[i].value)
            return
        if self.activation_type == 2:
            for i in range(len(self.neurons)):
                layer_derivative.neurons[i].value_activated_deriv = self.tanh_deriv(self.neurons[i].value)
            return


class neural_network:
    # each neural network has an array of neural layers and is responsible for training and making predictions
    def __init__(self, neural_layers, learning_rate):
        self.neural_layers = neural_layers
        self.output_layer = neural_layers[len(neural_layers) - 1]
        self.learning_rate = learning_rate

    def predict(self, input_vector):
        # activate layer by layer using activation functions, and a weighted sum of inputs and a bias to predict the output layer
        # index 0 is the input layer so start at first hidden layer
        self.set_input_layer(input_vector)
        value = 0
        for i in range(1, len(self.neural_layers)):
            for j in range(len(self.neural_layers[i].neurons)):
                for k in range(len(self.neural_layers[i].neurons[j].in_edges)):
                    value += self.neural_layers[i].neurons[j].in_edges[k].weight * self.neural_layers[i].neurons[j].in_edges[k].input_node.value
                self.neural_layers[i].neurons[j].value = value + self.neural_layers[i].neurons[j].bias
                value = 0
            self.neural_layers[i].layer_activation()
        prediction = []
        for i in range(len(self.output_layer.neurons)):
            prediction.append(self.output_layer.neurons[i].value_activated)
        return prediction

    def set_input_layer(self, input_vector):
        # takes an ordered set of input vectors and sets the input layer values
        for i in range(len(self.neural_layers[0].neurons)):
            self.neural_layers[0].neurons[i].value = input_vector[i]
            self.neural_layers[0].neurons[i].value_activated = input_vector[i]

    def train(self, input_vectors, targets, iterations):
        # adjust weights and biases then attempts to prune/grow the network
        cumulative_errors = []
        while (iterations > 0):
            # pick test data at random from the given test set
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            weights = self.back_propagation(input_vector, target)

            # Measure the cumulative error for all the instances
            if iterations % 100 == 0:
                cumulative_error = []
                for i in range(len(input_vectors)):
                    prediction = self.predict(input_vectors[i])
                    error = self.mean_squared_error(targets[i], prediction)
                    cumulative_error.append(error)
                cumulative_errors.append(np.sum(cumulative_error)/len(cumulative_error))
            iterations -= 1
        return cumulative_errors

    def back_propagation(self, input_vector, target):
        # weights array to be used in pruning and growth algorithm
        weights = []
        # forward pass
        output = self.predict(input_vector)
        # calculate loss gradient (derivative of the loss function)
        # loss function = mean squared error
        error_deriv = 2 * (output[0] - target[0])

        # calculate layer weights and biases for output layer
        cur_layer = self.output_layer
        cur_neuron = None
        cur_layer.layer_deriv()
        for i in range(len(cur_layer.neurons)):
            cur_neuron = cur_layer.neurons[i]
            # calculate derivative with respect to the node and gradient with respect to bias
            cur_neuron.node_deriv = error_deriv * cur_neuron.value_activated_deriv
            cur_neuron.bias_error = cur_neuron.node_deriv
            self.update_bias(cur_neuron)
            # calculate gradient with respect to weight for each input edge of the current node
            for j in range(len(cur_neuron.in_edges)):
                cur_edge = cur_neuron.in_edges[j]
                cur_edge.weight_error = cur_neuron.node_deriv * cur_edge.input_node.value_activated
                self.update_weight(cur_edge)
                weights.append(cur_edge.weight)

        # calculate layer weights and biases for hidden layers
        index = 2
        cur_layer = self.neural_layers[len(self.neural_layers) - index]
        while (cur_layer.layer_type != 1):
            cur_layer.layer_deriv()
            for i in range(len(cur_layer.neurons)):
                cur_neuron = cur_layer.neurons[i]
                # get weights and derivatives from previous layer calculations
                node_out_weights = []
                node_out_derivs = []
                for j in range(len(cur_neuron.out_edges)):
                    cur_edge = cur_neuron.out_edges[j]
                    node_out_weights.append(cur_edge.weight)
                    node_out_derivs.append(cur_edge.output_node.node_deriv)
                # calculate derivative with respect to the node and gradient with respect to bias
                cur_neuron.node_deriv = np.dot(node_out_weights, node_out_derivs) * cur_neuron.value_activated_deriv
                cur_neuron.bias_error = cur_neuron.node_deriv
                self.update_bias(cur_neuron)
                # calculate gradient with respect to weight for each input edge of the current node
                for j in range(len(cur_neuron.in_edges)):
                    cur_edge = cur_neuron.in_edges[j]
                    cur_edge.weight_error = cur_neuron.node_deriv * cur_edge.input_node.value_activated
                    self.update_weight(cur_edge)
                    weights.append(cur_edge.weight)
            index += 1
            cur_layer = self.neural_layers[len(self.neural_layers) - index]
        return weights

    def squared_error(self, target, prediction):
        return (target - prediction) ** 2

    def mean_squared_error(self, target, prediction):
        error = 0
        for i in range(len(target)):
            error += self.squared_error(target[i], prediction[i])
        error = error / len(target)
        return error

    def update_bias(self, neuron):
        neuron.bias = neuron.bias - (neuron.bias_error * self.learning_rate)

    def update_weight(self, edge):
        edge.weight = edge.weight - (edge.weight_error * self.learning_rate)