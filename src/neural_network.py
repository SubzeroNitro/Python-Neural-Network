from enum import Enum
from .activation_functions import *
from .error_functions import *
from .utilities import *
import numpy as np

# 'NeuralNetwork' class represents a simple feed forward neural network 
class NeuralNetwork:
    # Initializes the neural network with the specified shape
    # representing the number of layers and nodes per layer
    # an activation function and error function can be passed
    # to change which functions the neural network uses
    def __init__(self, shape: list, activation_function: ActivationFunction = ActivationFunction.SIGMOID, error_function: ErrorFunction = ErrorFunction.MEAN_SQUARED_ERROR):
        # Assert that the neural network has at least
        # two layers, at least one input and output,
        # and that the activation function is an
        # instance of the activation functions enum
        assert len(shape) >= 2
        assert shape[0] > 0
        assert shape[-1] > 0
        assert isinstance(activation_function, Enum)
        
        # Initialize the internally used activation
        # functions to the value passed by parameter
        self.__activation_function = activation_function.value[0]
        self.__activation_function_derivative = activation_function.value[1]
        self.__error_function = error_function.value[0]
        self.__error_function_derivative = error_function.value[1]
        
        # Initialize the shape of the neural network
        # layer by layer
        self.shape = shape
        self.num_inputs = shape[0]
        self.num_outputs = shape[-1]
        self.hidden_layers = shape[1:-1]
        
        # Initialize the weights and biases as well as
        # the internally used network activations and
        # outputs
        self.weights = []
        self.biases = []
        self.__network_activations = []
        self.__network_outputs = []

        # Set random weight values along all layers
        for i in range(len(shape) - 1):
            # Compute a vector of randomized weights
            layer_weight = random_weight((shape[i], shape[i + 1]))
            self.weights.append(layer_weight)
            
            # Biases are initially zero
            layer_biases = np.zeros((1, shape[i + 1]))
            self.biases.append(layer_biases)
            
            # Initializes the activations as vectors with empty
            # elements in the same shape as the network
            layer_activations = np.empty(shape[i])
            self.__network_activations.append(layer_activations)
            self.__network_outputs.append(layer_activations)
    
    # Compute the output of the network based
    # on a single set of inputs
    def compute_output(self, inputs: list):
        # Assert that the number of provided inputs
        # matches the number of network inputs
        assert len(inputs.flatten()) == self.num_inputs
        
        # The first layer outputs is equivalent to the input
        outputs = inputs
        
        # Loop through the layers of the network
        for i in range(len(self.shape) - 1):
            # Set the current layer activations as the previous output
            self.__network_activations[i] = outputs
            # The new output is the dot product of the previous output
            # and the weights of the current layer plus the current biases
            outputs = np.dot(outputs, self.weights[i]) + self.biases[i]
            
            # The current layer output before activation is now set to the 
            # new output
            self.__network_outputs[i] = outputs
            # The output is transformed via the activation function
            outputs = self.__activation_function(outputs)
        
        # The output is returned as a numpy array
        return np.array(outputs)
    
    # Trains the neural network by applying small
    # corrections to the weights and biases 
    def train(self, inputs: list, outputs: list, epochs: int, learning_rate: float):
        num_training_data = len(inputs)
    
        # Assert that the length of the input training
        # data is the same as the length of the output
        # training data
        assert num_training_data == len(outputs)
    
        # Train on the same dataset for the length of 'epochs'
        for i in range(1, epochs + 1):
            display_error = 0
        
            for j in range(num_training_data):
                # Compute the output of the network based on
                # a sample training input
                computed_output = self.compute_output(inputs[j])
                
                # Calculate how wrong the computed output is
                # using the error function
                display_error += self.__error_function(computed_output, outputs[j])
                error = self.__error_function_derivative(computed_output, outputs[j])

                # Loop backwards because this method of training
                # is known as backwards propogation where we begin
                # at the end and propogate corrections backwards
                for k in reversed(range(len(self.weights))):
                    # By calculating the derivative of the outputs
                    # with respect to the weight of the previous layer
                    # we find how much to adjust by
                    error *= self.__activation_function_derivative(self.__network_outputs[k])
                    updated_error = np.dot(error, self.weights[k].T)
                    
                    # The weight gradient is an amount that is calculated
                    # to correct the weights by
                    weight_gradient = np.dot(self.__network_activations[k].T, error)
                    
                    # Adjust the weights by the gradient multiplied by the
                    # learning rate to avoid overshooting the correction
                    self.weights[k] -= weight_gradient * learning_rate
                    # The correction for the bias is instead based on the
                    # error itself
                    self.biases[k] -= error * learning_rate
                    
                    # The error is 'propogated' to the next layer here
                    # for additional calculations to be performed
                    error = updated_error
            
            # Statistical information on the error over ever epoch
            print("Epoch " + str(i) + "/" + str(epochs) + "\t\tError: " + str(display_error / num_training_data))
    
    # Save the weights and biases to a file
    def save_model(self, path: str):
        file = open(path, "w")
        
        file.write(self.weights + "\n")
        file.write(self.biases)
    
    # Load weights and biases from a file
    def load_model(self, path: str):
        file = open(path, "r")
        
        self.weights = list(file.readline());
        self.biases = list(file.readline());
    
