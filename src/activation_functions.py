from enum import Enum
import numpy as np

# Several useful activation functions
# and their derivatives

def relu(x):
    return x if x >= 0 else 0

def relu_derivative(x):
    return x >= 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    value = sigmoid(x)
    return value - (value * value)
    
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    values = np.tanh(x)
    return 1 - (values * values)

def arctan(x):
    return np.arctan(x)

def arctan_derivative(x):
    return 1 / ((x * x) + 1)

# An enumeration that can be used to easily
# pass any of the four functions and their
# derivatives around
class ActivationFunction(Enum):
    RELU = relu, relu_derivative
    SIGMOID = sigmoid, sigmoid_derivative
    TANH = tanh, tanh_derivative
    ARCTAN = arctan, arctan_derivative