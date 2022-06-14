"""

The module for the ensemble
extended linear process.

"""
import math
from typing import Union, List, Optional, Dict

import torch
from torch import nn

from superTransformerLib.transformerLib import Glimpses


class Linear(nn.Module):
    """

    A Linear layer allowing head-dependent linear processing of data from shape
    to shape. JIT is supported as an instance.

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





    def __init__(self,
                 input_shape: Union[torch.Tensor, List[int], int],
                 output_shape: Union[torch.Tensor, List[int], int],
                 ensemble_shapes: Optional[Union[torch.Tensor, List[int], int]]=None):
        """

        :param input_shape: The shape of the input. May be an int, or a list/tuple of ints,
            or a tensor
        :param output_shape: The shape of the output. May be an int, or a list/tuple of ints,
            or a tensor
        :param ensemble_shapes: The size of the ensemble dimensions.
        :param ensemble_dims: The dimensions on which the ensemble is found.
        """
        # Super call

        super().__init__()

        # Implicit conversion

        if ensemble_shapes is None:
            ensemble_shapes = []
        elif isinstance(ensemble_shapes, int):
            ensemble_shapes = [ensemble_shapes]
        elif torch.is_tensor(ensemble_shapes) and ensemble_shapes.dim() == 0:
            ensemble_shapes = [ensemble_shapes]
        if isinstance(input_shape, int):
            input_shape = [input_shape]
        elif torch.is_tensor(input_shape) and input_shape.dim() == 0:
            input_shape = [input_shape]
        if isinstance(output_shape, int):
            output_shape = [output_shape]
        elif torch.is_tensor(output_shape) and output_shape.dim() == 0:
            output_shape = [output_shape]

        input_shape = torch.tensor(input_shape, dtype=torch.int64)
        output_shape = torch.tensor(output_shape, dtype=torch.int64)
        head_shapes = torch.tensor(ensemble_shapes, dtype=torch.int64)

        # Create kernel and bias. These include head dimensions if provided.

        if head_shapes is not None:

            kernel_shape = [*head_shapes, output_shape.prod(),input_shape.prod()]
            bias_shape = [*head_shapes, output_shape.prod()]
        else:
            kernel_shape = [output_shape.prod(), input_shape.prod()]
            bias_shape = [output_shape.prod()]

        kernel = torch.zeros(kernel_shape, requires_grad=True)
        kernel = torch.nn.init.kaiming_uniform_(kernel, a=math.sqrt(5))

        bias = torch.zeros(bias_shape, requires_grad=True)
        bias = torch.nn.init.zeros_(bias)

        # Store shapes and kernels

        self._input_shape = input_shape
        self._output_shape = output_shape

        self._kernel = nn.Parameter(kernel)
        self._bias = nn.Parameter(bias)
    def forward(self, tensor):

        # Flatten the relevent dimensions

        tensor = Glimpses.reshape(tensor, self._input_shape, int(self._input_shape.prod()))

        # Perform primary processing. Add an extra dimension on the end
        # of the input tensor to handle the matrix multiply, perform
        # matrix multiply, then add bias

        tensor = tensor.unsqueeze(-1)
        tensor = self._kernel.matmul(tensor)
        tensor = tensor.squeeze(-1)
        tensor = tensor + self._bias

        # Restore the dimensions, then return
        tensor = Glimpses.reshape(tensor, int(self._output_shape.prod()), self._output_shape)
        return tensor


class NamedLinear(nn.Module):
    """

    A Linear layer allowing head-dependent linear processing
    of incoming information by means of named tensors.



    Letting the head_shape parameter be none will disable it, resulting in broadcasting. Input
    shape, output shape, and head_shapes may all be just an integer, in which case it is
    assumed only a single dimension is involved.

    """





    def __init__(self,
                 input_shape: Dict[str, int],
                 output_shape: Dict[str, int],
                 ensemble_shapes: Dict[str, int]):
        """

        :param input_shape: The shape of the input, and the input names. The class
            will isolate these and flatten them for feeding through a linear process
        :param output_shape: The shape of the output. May be an int, or a list/tuple of ints,
            or a tensor.
        :param ensemble_shapes: The names and dimensions of the ensemble we are dealing with.
        """
        # Super call

        super().__init__()

        input_shape = torch.tensor(input_shape, dtype=torch.int64)
        output_shape = torch.tensor(output_shape, dtype=torch.int64)
        head_shapes = torch.tensor(ensemble_shapes, dtype=torch.int64)

        # Create kernel and bias. These include head dimensions if provided.

        if head_shapes is not None:

            kernel_shape = [*head_shapes, output_shape.prod(),input_shape.prod()]
            bias_shape = [*head_shapes, output_shape.prod()]
        else:
            kernel_shape = [output_shape.prod(), input_shape.prod()]
            bias_shape = [output_shape.prod()]

        kernel = torch.zeros(kernel_shape, requires_grad=True)
        kernel = torch.nn.init.kaiming_uniform_(kernel, a=math.sqrt(5))

        bias = torch.zeros(bias_shape, requires_grad=True)
        bias = torch.nn.init.zeros_(bias)

        # Store shapes and kernels

        self._input_shape = input_shape
        self._output_shape = output_shape

        self._kernel = nn.Parameter(kernel)
        self._bias = nn.Parameter(bias)
    def forward(self, tensor):


        # Perform primary processing. Add an extra dimension on the end
        # of the input tensor to handle the matrix multiply, perform
        # matrix multiply, then add bias
        shape = tensor.shape[:-len(self._input_shape)]
        tensor = tensor.flatten(0, -len(self._input_shape)-1)

        tensor = tensor.unsqueeze(-1)
        tensor = self._kernel.matmul(tensor)
        tensor = tensor.squeeze(-1)
        tensor = tensor + self._bias

        tensor = tensor.unflatten(dim=0, sizes=shape)
        return tensor