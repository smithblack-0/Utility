

# perform imports

import numpy as np
import torch
from torch import nn

import math
import numbers

#perform library imports
from Utility.Torch import Glimses

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
        #Super call

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

        self._preprocessor = lambda x: Glimses.view(x, input_shape, np.prod(input_shape))
        self._postprocesser = lambda x: Glimses.view(x, np.prod(output_shape), output_shape)

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

        self._kernel = kernel
        self._bias = bias

    def forward(self, tensor):

        # Flatten the relavent dimensions

        tensor = self._preprocessor(tensor)

        # Perform primary processing. Add an extra dimension on the end
        # of the input tensor to handle the matrix multiply, perform
        # matrix multiply then add bias

        tensor = tensor.unsqueeze(-1)
        tensor = self._kernel.matmul(tensor)
        tensor = tensor.squeeze(-1)
        tensor = tensor + self._bias

        # Restore the dimensions
        tensor = self._postprocesser(tensor)

        # Return
        return tensor


class Transformer(nn.Module):
    __permitted = (None, "lower", "upper")

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        assert value in self.__permitted, "mask cannot be set to this"
        self._mask = value

    def __init__(self, channel_dim, head_width, mask=None):

        """

        Accepted mask is "lower", "upper", or none

        """

        # Spin up torch
        super().__init__()

        # Create action generators
        QueryGen = LinearReshape(channel_dim, (head_width, channel_dim))
        KeyGen = LinearReshape(channel_dim, (head_width, channel_dim))
        ValGen = LinearReshape(channel_dim, (head_width, channel_dim))

        CollapseGen = LinearReshape((head_width, channel_dim), channel_dim)

        # Create actions. Note the swap is needed to get the head in front of the items.

        self._query = lambda x: QueryGen(x).swapdims(-2, -3)
        self._key = lambda x: KeyGen(x).swapdims(-2, -3)
        self._value = lambda x: ValGen(x).swapdims(-2, -3)
        self._dehead = lambda x: CollapseGen(x.transpose(-2, -3))

        self.mask = mask

    def forward(self, query, content, mask=None):
        # Create query, key, value

        query = self._query(query)
        key = self._key(content).swapdims(-1, -2)
        value = self._value(content)

        # Create focus matrix. Mask. Softmax.

        focus = query.matmul(key)
        focus_dims = focus.shape[-2:]
        if mask is None:
            # Runs only if not provided a mask.
            if self.mask == "lower":
                mask = torch.tril(torch.ones(focus_dims))
                focus = focus.masked_fill(mask == 0, -1e9)
            if self.mask == "upper":
                mask = torch.triu(torch.ones(focus_dims))
                focus = focus.masked_fill(mask == 0, -1e9)

        focus = F.softmax(focus, dim=-1)

        # Apply focus matrix to values. Then compact head

        output = focus.matmul(value)
        output = self._dehead(output)

        return output

