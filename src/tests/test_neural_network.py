from ..neural_network import NeuralNetwork
import numpy as np
import unittest

# Test cases related to the main Neural Network class
class NeuralNetworkTest(unittest.TestCase):
    # Test whether the members in NeuralNetwork are correctly initialized
    def when_initialized_should_create_neural_network_successfully(self):
        # Initialize shape to random length
        shape = []
        num_layers = np.random.randint(2, 10)

        # Set the length of each layer to random values
        for _ in range(num_layers):
            shape.append(np.random.randint(1, 100))

        # Create Neural Network
        nn = NeuralNetwork(shape)

        # Test that the number of internal layers, inputs,
        # and outputs matches the expected values
        self.assertEqual(len(nn.shape), num_layers)
        self.assertEqual(nn.num_inputs, shape[0])
        self.assertEqual(nn.num_outputs, shape[-1])

        # Loop between all of the hidden layers
        for i in range(num_layers - 1):
            # Get the length of the weight matrix at
            # the current index and assert that the
            # length matches the expected value within
            # the shape
            layer_length = len(nn.weights[i])
            self.assertEqual(layer_length, shape[i])

            # For every set of weights at the current layer
            # test that the number of weights matches the
            # number of nodes in the next layer
            for j in range(layer_length):
                self.assertEqual(len(nn.weights[i][j]), shape[i + 1])
            
            # Test that the number of biases in the current
            # layer matches the number of nodes in the next layer
            self.assertEqual(len(nn.biases[i][0]), shape[i + 1])

    # Test that the compute_output method works in a small network
    def when_expected_input_is_passed_into_small_network_should_compute_output(self):     
        # Create neural network with shape
        shape = [ 2, 3, 3, 2 ]
        nn = NeuralNetwork(shape)
        
        # Create an array of inputs for testing
        inputs = np.array([[ 1, 1 ]])
        # Call the 'compute_output' method with the inputs
        outputs = nn.compute_output(inputs).flatten()
        
        # Assert that the output is of type numpy.ndarray
        self.assertIsInstance(outputs, np.ndarray)
        # Assert that the number of outputs is the expected value 
        self.assertEqual(len(outputs), shape[-1])
        
        # Loop through each output
        for output in outputs:
            # Assert that each output is of type float
            self.assertIsInstance(output, float)
            # Assert that each output is a valid number
            self.assertFalse(np.isnan(output))