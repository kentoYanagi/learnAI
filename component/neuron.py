from utils import sigmoid

# Neuron
class Neuron:
    def __init__(self):
        self.input_sum = 0.0
        self.output = 0.0

    def set_input(self, input_value):
        self.input_sum += input_value
    
    def get_output(self):
        self.output = sigmoid(self.input_sum)
        return self.output

if __name__ == "__main__":
    neuron = Neuron()
    neuron.set_input(1)
    print(neuron.get_output())