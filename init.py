import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from component.neural_network import NeuralNetwork

if __name__ == "__main__":
    iris = datasets.load_iris()

    iris_data = iris.data

    sepal_length = iris_data[:100, 0]
    sepal_width = iris_data[:100, 1]

    sepal_length_avg = np.average(sepal_length)
    sepal_length -= sepal_length_avg

    sepal_width_avg = np.average(sepal_width)
    sepal_width -= sepal_width_avg

    setosa_or_versicolor = [[sepal_length[i], sepal_width[i]] for i in range(100)]
    neural_network = NeuralNetwork()

    predicted_setosa = [[], []]
    predicted_versicolor = [[], []]
    for data in setosa_or_versicolor:
        if neural_network.commit(data) < 0.5:
            predicted_setosa[0].append(data[0] + sepal_length_avg)
            predicted_setosa[1].append(data[1] + sepal_width_avg)
        else:
            predicted_versicolor[0].append(data[0] + sepal_length_avg)
            predicted_versicolor[1].append(data[1] + sepal_width_avg)

    plt.scatter(predicted_setosa[0], predicted_setosa[1], label='Setosa')
    plt.scatter(predicted_versicolor[0], predicted_versicolor[1], label='Versicolor')
    plt.legend()
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Sepal width (cm)")
    plt.show()