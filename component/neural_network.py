from .neuron import Neuron

# ニューラルネットワーク

class NeuralNetwork:
    def __init__(self):
        self.neuron = Neuron()
        # 重み
        # self.w = [0.5, -0.2]
        self.input_to_middle_weight = [[4.0, 4.0], [4.0, 4.0]]
        self.middle_to_output_weight = [[1.0, -1.0]]

        # バイアス
        self.middle_bias = [2.0, -2.0]
        self.output_bias = [-0.5]

        # 各層の手動宣言
        self.input_layer = [0.0, 0.0]
        self.middle_layer = [Neuron(), Neuron()]
        self.output_layer = [Neuron()]

    def commit(self, input_data):
        # 全ての層のリセット
        for index in range(len(self.input_layer)):
            self.input_layer[index] = input_data[index]

        for index in range(len(self.middle_layer)):
            self.middle_layer[index].reset()
        self.output_layer[0].reset()

        # 入力層 → 中間層
        for index in range(len(self.middle_layer)):
            self.middle_layer[index].set_input(self.input_layer[0] * self.input_to_middle_weight[index][0])
            self.middle_layer[index].set_input(self.input_layer[1] * self.input_to_middle_weight[index][1])
            self.middle_layer[index].set_input(self.middle_bias[index])

        self.output_layer[0].set_input(self.middle_layer[0].get_output() * self.middle_to_output_weight[0][0])
        self.output_layer[0].set_input(self.middle_layer[1].get_output() * self.middle_to_output_weight[0][1])
        self.output_layer[0].set_input(self.output_bias[0])
        
        return  self.output_layer[0].get_output()

if __name__ == "__main__":
    from random import randint
    neural_network = NeuralNetwork()
    result = neural_network.commit([randint(0, 10) for _ in range(3)])
    print(result)