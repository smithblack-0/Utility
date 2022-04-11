"""

The Capture module.

This contains classes for capturing information about
gradients on sparse-dense transition points. In particular,

"""

import torch
import torch_sparse
from torch import nn

class Capture(nn.Module):
    """

    The base capture class for capturing gradients

    A capture class should be initialize once at the beginning
    of the
    """

    def __init__(self):
        super().__init__()



class Matmul(Connectivity):
    """
    The matmul capture unit.

    Accepts sparse tensors, and matmul's them together.
    On the backwards pass, captures the K highest gradients.

    This only works on transitions between sparse and dense
    """
    def __init__(self, number_of_captures: int,
                 starting_percentage = 0.1,
                 change_aggressiveness=0.1,
                 capture_enabled: bool = True):
        super().__init__()
    def forward(self, sparse, dense):
        if isinstance(matrix_a, torch_sparse.SparseTensor) and isinstance(matrix_b, torch_sparse.SparseTensor):
            #sparse-sparse
            #If a row





class Pull(nn.Module)
    """
    
    The basic retrieval unit.
    
    This is expected to be activated after all
    other processing is done, in something like 
    an optimizer.
    
    
    """