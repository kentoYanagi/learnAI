import numpy 

def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))

if __name__ == '__main__':
    print(sigmoid(2))