# Caleb Smith
# 01/13/2023
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from deeplearning import NeuralNetwork
from deeplearning import NeuralLayer

def main():
    input_vectors = np.array(
        [
            [3, 4, 1],
            [4, 5, 0],
            [-2, 3, 1],
            [3, 1, 1],
            [2, 0, 10],
            [-100, 7, 1],
            [1, 1, 0],
        ]
    )

    targets = np.array([1, 1, 1, -1, 1, -1, 0])

    learning_rate = 0.01

    neural_layer1 = NeuralLayer(2, 3)

    neural_layer2 = NeuralLayer(1, 1)

    neural_layer3 = NeuralLayer(3, 1)

    layers = [neural_layer1, neural_layer3]

    neural_network = NeuralNetwork(learning_rate, layers)

    input_vector = [3, 1, 1]
    print(neural_network.predict(input_vector))

    training_error = neural_network.train(input_vectors, targets, 10000)

    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.show()

    input_vector = [3, 1, 1]
    print(neural_network.predict(input_vector))

if __name__ == "__main__":
    main()