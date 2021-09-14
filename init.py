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
    sepal_length -= sepal_length_avg # 平均を引くことで、全データを平均値からの距離を割り出す。

    sepal_width_avg = np.average(sepal_width)
    sepal_width -= sepal_width_avg

    # [訓練データ1, 訓練データ2, 教師データ]
    setosa_or_versicolor = [[sepal_length[i], sepal_width[i], iris.target[i]] for i in range(100)]
    neural_network = NeuralNetwork()

    print("before ---------------------")
    print(neural_network.input_to_middle_weight)
    print(neural_network.middle_bias)
    print(neural_network.middle_to_output_weight)
    print(neural_network.output_bias)
    neural_network.commit(setosa_or_versicolor[0][:2])
    neural_network.train(setosa_or_versicolor[0][2])
    print("after ---------------------")
    print(neural_network.input_to_middle_weight)
    print(neural_network.middle_bias)
    print(neural_network.middle_to_output_weight)
    print(neural_network.output_bias)

    predicted_setosa = [[], []]
    predicted_versicolor = [[], []]
    for data in setosa_or_versicolor:
        if neural_network.commit(data[:2]) < 0.5:
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