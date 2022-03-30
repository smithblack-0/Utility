# perform imports

import numpy as np
import torch
from torch import nn

import math

# perform library imports
from Utility.Torch import Glimpses





### Head accommodation on the linear layer ###
class Linear(nn.Module):
    """

    A Linear layer allowing head-dependent linear processing of data from shape
    to shape.

    An instance is made by providing a list of head_shapes,
    an input_shape tuple, an output_shape tuple.

    This is then used to initialize a head dependent linear remap
    from input shape to output shape. That will then be accessed
    through the instance call

    It is expected that the input format will be in the form of

    [..., heads, input_shape]

    Returning something of format

    [..., heads, output_shape]


    Letting the head_shape parameter be none will disable it, resulting in broadcasting. Input
    shape, output shape, and head_shapes may all be just an integer, in which case it is
    assumed only a single dimension is involved.

    """

    def __init__(self, input_shape, output_shape, head_shapes=None):
        """

        :param input_shape: The shape of the input. May be an int, or a list/tuple of ints
        :param output_shape: The shape of the output. May be an int, or a list/tuple of ints.
        :param head_shapes: The head dimensions, which come immediately prior to the
            input dimensions. May be an int, or a list/tuple of ints.
        """
        # Super call

        super().__init__()

        # Implicit conversion
        if head_shapes is None:
            head_shapes = []
        elif isinstance(head_shapes, int):
            head_shapes = [head_shapes]
        if isinstance(input_shape, int):
            input_shape = [input_shape]
        if isinstance(output_shape, int):
            output_shape = [output_shape]

        # Create preprocesser and postprocessor. These flatten, and unflatten, the
        # dimensions we care about

        self._preprocessor = lambda x: Glimpses.view(x, input_shape, np.prod(input_shape))
        self._postprocesser = lambda x: Glimpses.view(x, np.prod(output_shape), output_shape)

        # Create kernel and bias. These include head dimensions if provided.

        if head_shapes is not None:
            kernel_shape = [*head_shapes, np.prod(output_shape), np.prod(input_shape)]
            bias_shape = [*head_shapes, np.prod(output_shape)]
        else:
            kernel_shape = [np.prod(output_shape), np.prod(input_shape)]
            bias_shape = [np.prod(output_shape)]

        kernel = torch.zeros(kernel_shape, requires_grad=True)
        kernel = torch.nn.init.kaiming_uniform_(kernel, a=math.sqrt(5))

        bias = torch.zeros(bias_shape, requires_grad=True)
        bias = torch.nn.init.zeros_(bias)

        # Store

        self._kernel = nn.Parameter(kernel)
        self._bias = nn.Parameter(bias)

    def forward(self, tensor):

        # Flatten the relevent dimensions

        tensor = self._preprocessor(tensor)

        # Perform primary processing. Add an extra dimension on the end
        # of the input tensor to handle the matrix multiply, perform
        # matrix multiply, then add bias

        tensor = tensor.unsqueeze(-1)
        tensor = self._kernel.matmul(tensor)
        tensor = tensor.squeeze(-1)
        tensor = tensor + self._bias

        # Restore the dimensions
        tensor = self._postprocesser(tensor)

        # Return
        return tensor



