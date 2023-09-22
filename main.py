from neuralnetwork import NeuralNetwork
from neuralnetwork import ActivationFunction
from neuralnetwork import ErrorFunction

from keras.datasets import mnist
from keras.utils import to_categorical

import numpy as np

neuralnetwork = NeuralNetwork([ 784, 100, 50, 10 ], ActivationFunction.SIGMOID, ErrorFunction.MEAN_SQUARED_ERROR)

(inputs, outputs), (test_inputs, test_outputs) = mnist.load_data()

inputs = inputs.reshape(inputs.shape[0], 1, 784)
inputs = inputs.astype("float32")
inputs /= 255
outputs = to_categorical(outputs)

test_inputs = test_inputs.reshape(test_inputs.shape[0], 1, 784)
test_inputs = test_inputs.astype("float32")
test_inputs /= 255
test_outputs = to_categorical(test_outputs)

def main():
    neuralnetwork.train(inputs[0:1000], outputs[0:1000], 10, 0.1)
    
    np.set_printoptions(suppress = True)
    
    print(test_inputs[0])
    
    for i in range(5):
        test = neuralnetwork.compute_output(test_inputs[i])
        
        print("\nOutput Value: ")
        print(np.round(test, 2).flatten())
        print("Actual Value: ")
        print(test_outputs[i])

if __name__ == "__main__":
    main()
