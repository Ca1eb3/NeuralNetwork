# Caleb Smith
# 02/13/2023
import neuralnet as dl
class neuralnetbuilder:
    def __init__(self):
        self.neural_layers = []
        self.learning_rate = None

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_neural_layers(self, num_layers, num_nodes_layers, layer_activation_types):
        neurons_cur = []
        for i in range(num_layers):
            for j in range(num_nodes_layers[i]):
                neuron = neuronbuilder().build()
                neurons_cur.append(neuron)
            if i == 0:
                input_layer = neurallayerbuilder()
                input_layer.set_activation_type(layer_activation_types[i])
                input_layer.set_layer_type(1)
                input_layer.set_neurons(neurons_cur.copy())
                self.neural_layers.append(input_layer.build())
                neurons_cur.clear()
            elif i == num_layers - 1:
                output_layer = neurallayerbuilder()
                output_layer.set_activation_type(layer_activation_types[i])
                output_layer.set_layer_type(3)
                output_layer.set_neurons(neurons_cur.copy())
                self.neural_layers.append(output_layer.build())
                neurons_cur.clear()
                for j in range(len(self.neural_layers[i].neurons)):
                    for k in range(len(self.neural_layers[i - 1].neurons)):
                        #self.neural_layers[i].neurons[j] # output node
                        #self.neural_layers[i - 1].neurons[k] # input node
                        edge = edgebuilder()
                        edge.set_input_node(self.neural_layers[i - 1].neurons[k])
                        edge.set_output_node(self.neural_layers[i].neurons[j])
                        edge.build()
            else:
                hidden_layer = neurallayerbuilder()
                hidden_layer.set_activation_type(layer_activation_types[i])
                hidden_layer.set_layer_type(2)
                hidden_layer.set_neurons(neurons_cur.copy())
                self.neural_layers.append(hidden_layer.build())
                neurons_cur.clear()
                for j in range(len(self.neural_layers[i].neurons)):
                    for k in range(len(self.neural_layers[i - 1].neurons)):
                        #self.neural_layers[i].neurons[j] # output node
                        #self.neural_layers[i - 1].neurons[k] # input node
                        edge = edgebuilder()
                        edge.set_input_node(self.neural_layers[i - 1].neurons[k])
                        edge.set_output_node(self.neural_layers[i].neurons[j])
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