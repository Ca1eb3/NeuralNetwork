# Caleb Smith
# 01/13/2023
from matplotlib import pyplot as plt
import numpy as np
import math
from enum import Enum

class ActivationFunction(Enum):
    SIGMOID = 1
    DOTPRODUCT = 2
    TANH = 3

class NeuralNetwork:
    def __init__(self, learning_rate, neural_layers):
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.neural_layers = np.array(neural_layers)

    def predict(self, input_vector):
        prediction = self.neural_layers[0]._layer_activation(input_vector)
        for i in range(0, len(prediction)):
            prediction[i] = prediction[i] + self.bias
        for i in range(1, len(self.neural_layers)):
            prediction = self.neural_layers[i]._layer_activation(prediction)
        return prediction

    def _compute_gradients(self, input_vector, target):
        derror_dweights = []
        layer_prediction = []
        prediction = self.neural_layers[0]._layer_activation(input_vector)
        for i in range(0, len(prediction)):
            prediction[i] = prediction[i] + self.bias
        prediction = np.array(prediction)
        layer_prediction.append(prediction)
        for i in range(1, len(self.neural_layers)):
            prediction = self.neural_layers[i]._layer_activation(prediction)
            prediction = np.array(prediction)
            layer_prediction.append(prediction)
        layer_prediction = np.array(layer_prediction)

        # determine error in prediction
        derror_dprediction = 2 * (prediction - target)

        # calculate derivative of bias and weights for input layer
        dprediction_dlayer1 = self.neural_layers[0]._layer_deriv(layer_prediction[0])
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.neural_layers[0].weights) + (1 * input_vector)
        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights_layer1 = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )
        derror_dweights.append(derror_dweights_layer1)

        # calculate derivative of weights for hidden layers
        for i in range(1, len(self.neural_layers) - 1):
             dprediction_dlayer = self.neural_layers[i]._layer_deriv(layer_prediction[i])
             dlayer_dweights = (0 * self.neural_layers[i].weights) + (1 * input_vector)
             derror_dweights_layer = (
                    derror_dprediction * dprediction_dlayer * dlayer_dweights
             )
             derror_dweights.append(derror_dweights_layer)

        # calculate derivative of weights for output layer
        dprediction_dlayer_final = self.neural_layers[len(self.neural_layers) - 1]._layer_deriv(layer_prediction[len(self.neural_layers) - 1])
        dlayer_final_dweights = (0 * self.neural_layers[len(self.neural_layers) - 1].weights) + (1 * input_vector)
        derror_dweights_layer_final = (
             derror_dprediction * dprediction_dlayer_final * dlayer_final_dweights
        )
        derror_dweights.append(derror_dweights_layer_final)
        
        return derror_dbias, derror_dweights


    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        for i in range(0, len(derror_dweights)):
            self.neural_layers[i].weights = self.neural_layers[i].weights - ( 
                derror_dweights[i] * self.learning_rate 
            )

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

class NeuralLayer:
    def __init__(self, activation_type, input_size):
        list_weights = []
        for i in range(0, input_size):
            list_weights.append(np.random.randn())
        self.weights = np.array(list_weights)
        self.activation_type = activation_type

    def _sigmoid(self, input_array):
        for x in range(0, len(input_array)):
            input_array[x] = 1 / (1 + np.exp(-input_array[x]))
        return input_array

    def _sigmoid_deriv(self, input_array):
        for x in range(0, len(input_array)):
            input_array[x] = self._sigmoid(np.array([input_array[x]])) * (1 - self._sigmoid(np.array([input_array[x]])))
        return np.array([input_array])

    def _dot_product(self, input_array):
        return np.array([np.dot(input_array, self.weights)])

    def _dot_product_deriv(self):
        return np.array(1)

    def _tanh(self, input_array):
        for x in range(0, len(input_array)):
            input_array[x] = (np.exp(input_array[x])-np.exp(-input_array[x]))/(np.exp(input_array[x])+np.exp(-input_array[x]))
        return input_array

    def _tanh_deriv(self, input_array):
        for x in range(0, len(input_array)):
            input_array[x] = 1 - (self._tanh(np.array([input_array[x]])) ** 2)
        return input_array

    def _layer_activation(self, input_array):
        if self.activation_type == 1:
            return self._sigmoid(input_array)
        if self.activation_type == 2:
            return self._dot_product(input_array)
        if self.activation_type == 3:
            return self._tanh(input_array)

    def _layer_deriv(self, input_array):
        if self.activation_type == 1:
            return self._sigmoid_deriv(input_array)
        if self.activation_type == 2:
            return self._dot_product_deriv()
        if self.activation_type == 3:
            return self._tanh_deriv(input_array)

