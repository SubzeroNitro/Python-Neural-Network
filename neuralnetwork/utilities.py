import numpy as np

# Determines an appropriate limit for the random
# initialization of a weight
def glorot_initialization_limit(num_inputs: int) -> float:
    return 1.0 / np.sqrt(num_inputs)

def random_weight(shape: list) -> list:
    # Calculate a limit and then apply a random uniform
    # distribution within the limits according to the
    # size of the shape
    limit = glorot_initialization_limit(shape[0])
    weight = np.random.uniform(-limit, limit, shape)
    
    return weight
