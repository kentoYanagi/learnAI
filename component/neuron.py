
# Neuron
class Neuron:
    def __init__(self):
        self.input_sum = 0.0

    def set_input(self, input_value):
        self.input_sum += input_value
        print(self.input_sum)

if __name__ == "__main__":
    Neuron().set_input(1)