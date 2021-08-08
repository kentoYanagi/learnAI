from neuron import Neuron

# ニューラルネットワーク

class NeuralNetwork:
    def __init__(self):
        self.neuron = Neuron()

    def commit(self, input_data):
        for data in input_data:
            self.neuron.set_input(data)

if __name__ == "__main__":
    from random import randint
    NeuralNetwork().commit([randint(0, 10) for _ in range(10)])
    