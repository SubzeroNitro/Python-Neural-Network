from neuralnetwork import NeuralNetwork
import numpy as np
import unittest

class NeuralNetworkTest(unittest.TestCase):
    # Test whether the members in NeuralNetwork are
    # correctly initialized
    def test_init(self):
        # Test 50 times
        for _ in range(50):
            shape = []
            num_layers = np.random.randint(2, 10)

            for _ in range(num_layers):
                shape.append(np.random.randint(1, 100))
    
            nn = NeuralNetwork(shape)

            self.assertEqual(len(nn.shape), num_layers)
            self.assertEqual(nn.num_inputs, shape[0])
            self.assertEqual(nn.num_outputs, shape[-1])

            for i in range(num_layers - 1):
                layer_length = len(nn.weights[i])
                self.assertEqual(layer_length, shape[i])

                for j in range(layer_length):
                    self.assertEqual(len(nn.weights[i][j]), shape[i + 1])
                
                self.assertEqual(len(nn.biases[i][0]), shape[i + 1])

    def test_compute_output(self):     
        shape = [ 2, 3, 3, 2 ]
        nn = NeuralNetwork(shape)
        
        inputs = np.array([[ 1, 1 ]])
        outputs = nn.compute_output(inputs).flatten()
        
        self.assertTrue(isinstance(outputs, np.ndarray))
        self.assertEqual(len(outputs), shape[-1])
        
        for output in outputs:
            self.assertTrue(isinstance(output, float))
            self.assertFalse(np.isnan(output))
        
if __name__ == "__main__":
    NeuralNetworkTest.main()
