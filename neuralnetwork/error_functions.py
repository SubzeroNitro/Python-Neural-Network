from multipledispatch import dispatch as overload
from collections.abc import Iterable
from enum import Enum
import numpy as np

# In this case there is only one error function
# provided, however, it would be trivial to add
# more functions such as Cross Entropy.

def mean_squared_error(value: float, target: float) -> float:
    difference = target - value
    return difference * difference
    
def mean_squared_error(values: Iterable, targets: Iterable) -> Iterable:
    differences = targets - values
    return np.mean(differences * differences)

def mean_squared_error_derivative(value: float, target: float) -> float:
    return 2 * (value - target)
    
def mean_squared_error_derivative(values: Iterable, targets: Iterable) -> Iterable:
    return 2 * (values - targets) / len(values)

# An enumeration of error functions which packages the
# function and its derivative
class ErrorFunction(Enum):
    MEAN_SQUARED_ERROR = mean_squared_error, mean_squared_error_derivative
