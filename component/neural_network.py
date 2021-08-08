from neuron import Neuron

# ニューラルネットワーク

class NeuralNetwork:
    def __init__(self):
        self.neuron = Neuron()
        self.w = [1.5, 0.75, -1.0]
        self.bias = 1.0

    def commit(self, input_data):
        for index in range(3):
            data = input_data[index] * self.w[index] # 重みの計算
            self.neuron.set_input(data)
        self.neuron.set_input(self.bias)
        return self.neuron.get_output()

if __name__ == "__main__":
    from random import randint
    neural_network = NeuralNetwork()
    result = neural_network.commit([randint(0, 10) for _ in range(3)])
    print(result)

    
    