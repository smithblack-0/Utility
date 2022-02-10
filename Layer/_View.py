
#Library imports
import torch
from torch import nn
import numpy as np

#import functions
import Functional as S_F

class View(nn.Module):
    """

    A viewing class. Stores the reshape
    then applies it over and over again. Notably,
    will only reshape the two provided dimensions. These
    dimensions must contain the same number of tensor entries

    For instance, given input_shape (4,2), output_shape (8), then given
    given tensor of shape (4, 3, 4,2), the result is (4, 3, 8)

    Expects input_shape, output_shape to be either an int or a list
    of ints. Expects the layer call to be a tensor.
    """
    def __init__(self, input_shape, output_shape):
        #Startup torch
        super().__init__()

        #Store shapes
        self._input_shape = input_shape
        self._output_shape = output_shape
    def forward(self, tensor):
        return S_F.View(tensor, self._input_shape, self._output_shape)
