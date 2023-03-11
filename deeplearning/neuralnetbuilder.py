# Caleb Smith
# 02/13/2023
import neuralnet as dl
class neuralnetbuilder:
    def __init__(self):
        self.neural_layers = []
        self.learning_rate = None

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_neural_layers(self, num_nodes_input_layer, num_nodes_output_layer, num_hidden_layers, num_nodes_hidden_layers, hidden_layer_activation_types, output_layer_activation_type):
        # build input layer
        neurons = []
        for i in range(num_nodes_input_layer):
            neuron = neuronbuilder().build()
            neurons.append(neuron)
        input_layer = neurallayerbuilder()
        input_layer.set_neurons(neurons)
        input_layer.set_activation_type(0)
        input_layer.set_layer_type(1)
        self.neural_layers.append(input_layer.build())
        
        # build hidden layers
        neurons.clear()
        for i in range(num_hidden_layers):
            hidden_layer = neurallayerbuilder()
            for j in range(num_nodes_hidden_layers[i]):
                neuron = neuronbuilder().build()
                neurons.append(neuron)
            hidden_layer.set_neurons(neurons)
            hidden_layer.set_activation_type(hidden_layer_activation_types[i])
            hidden_layer.set_layer_type(2)
            self.neural_layers.append(hidden_layer.build())
            for j in range(len(self.neural_layers[i].neurons)):
                edge = edgebuilder()
                edge.set_input_node(self.neural_layers[i].neurons[j])
                edge.set_output_node(self.neural_layers[i + 1].neurons[0])
                edge.build()  
            neurons.clear()

        # build output layer
        for i in range(num_nodes_output_layer):
            neuron = neuronbuilder().build()
            neurons.append(neuron)
        output_layer = neurallayerbuilder()
        output_layer.set_neurons(neurons)
        output_layer.set_activation_type(output_layer_activation_type)
        output_layer.set_layer_type(3)
        self.neural_layers.append(output_layer.build())
        for i in range(len(self.neural_layers[len(self.neural_layers) - 1].neurons)):
            edge = edgebuilder()
            edge.set_input_node(self.neural_layers[len(self.neural_layers) - 2].neurons[0])
            edge.set_output_node(self.neural_layers[len(self.neural_layers) - 1].neurons[i])
            edge.build()

    def build(self):
        return dl.neural_network(self.neural_layers, self.learning_rate)

class neurallayerbuilder:
    def __init__(self):
        self.activation_type = None
        self.layer_type = None
        self.neurons = []

    def set_activation_type(self, activation_type):
        self.activation_type = activation_type

    def set_layer_type(self, layer_type):
        self.layer_type = layer_type

    def set_neurons(self, neurons):
        for i in range(len(neurons)):
            self.neurons.append(neurons[i])

    def build(self):
        return dl.neural_layer(self.activation_type, self.neurons, self.layer_type)

class neuronbuilder:
    def __init__(self):
        None

    def build(self):
        return dl.neuron()

class edgebuilder:
    def __init__(self):
        self.input_node = None
        self.output_node = None

    def set_input_node(self, input_node):
        self.input_node = input_node

    def set_output_node(self, output_node):
        self.output_node = output_node

    def build(self):
        edge = dl.edge(self.input_node, self.output_node)
        edge.set_node_edges()
        return edge