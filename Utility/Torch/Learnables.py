# perform imports
from typing import Union, Sequence, Optional, Callable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import math

# perform library imports
from Utility.Torch import Glimpses, Paddings


### Head accommodation on the linear layer ###
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
                 head_shapes: Optional[Union[torch.Tensor, List[int], int]]=None):
        """

        :param input_shape: The shape of the input. May be an int, or a list/tuple of ints,
            or a tensor
        :param output_shape: The shape of the output. May be an int, or a list/tuple of ints,
            or a tensor
        :param head_shapes: The head dimensions, which come immediately prior to the
            input dimensions. May be None, an int, or a list/tuple of ints, or a tensor
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

        input_shape = torch.tensor(input_shape, dtype=torch.int64)
        output_shape = torch.tensor(output_shape, dtype=torch.int64)
        head_shapes = torch.tensor(head_shapes, dtype=torch.int64)

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

        tensor = Glimpses.view(tensor, self._input_shape, int(self._input_shape.prod()))

        # Perform primary processing. Add an extra dimension on the end
        # of the input tensor to handle the matrix multiply, perform
        # matrix multiply, then add bias

        tensor = tensor.unsqueeze(-1)
        tensor = self._kernel.matmul(tensor)
        tensor = tensor.squeeze(-1)
        tensor = tensor + self._bias

        # Restore the dimensions, then return
        tensor = Glimpses.view(tensor, int(self._output_shape.prod()), self._output_shape)
        return tensor


class BandedMultiheadedAttention(nn.Module):
    """
    This layer performs banded attention in a memory efficient manner in
    pytorch

    Banded attention performs attention within, suprise suprise, an exclusive, windowed band.
    views are used to keep this a memory efficient operation.
    """

    def __init__(self,
                 d_model: int,
                 kernel_width: int,
                 heads: int = 5,
                 dilation_rates: Optional[List[int]] = None,
                 compression_ratio: Optional[Tuple[int, int]] = None,
                 ):
        """

        :param d_model: The width of the embeddings
        :param kernel_width: How wide to make the key-value extraction kernel
        :param heads: How many different heads to make
        :param dilation_rates: The dilation rate per head. MUST match head length if defined
        :param offsets: The offsets per head. MUST match head length if defined
        """

        #Start torch
        super().__init__()

        #assert

        assert isinstance(d_model, int)
        assert isinstance(kernel_width, int)
        assert isinstance(heads, int)
        assert isinstance(compression_ratio, tuple)
        assert len(compression_ratio) == 2
        assert isinstance(compression_ratio[0], int)
        assert isinstance(compression_ratio[1], int)

        assert d_model >= 1
        assert kernel_width >= 1
        assert heads >= 1

        if dilation_rates is None:
            dilation_rates = [1]*heads

        assert len(dilation_rates) == heads

        dilation_rates = torch.Tensor(dilation_rates)




        #Store persistant useful constants

        self.heads = heads
        self.query_compression = compression_ratio[0]
        self.content_compression = compression_ratio[1]

        self.query_kernel = compression_ratio[0]
        self.query_stride = compression_ratio[0]

        self.content_kernel = kernel_width*compression_ratio[1]
        self.content_stride = compression_ratio[1]

        self.dilation = dilation_rates
        self.offset = offsets

        # Create projection layers.

        d_kernel = d_model//heads

        self._Query = Linear([self.query_kernel, d_model], [self._query_kernel,d_kernel], heads)
        self._Key = Linear([self.content_kernel, d_model], [self._content_kernel, d_kernel], heads)
        self._Value = Linear([self._content_kernel, d_model], [self._content_kernel, d_kernel], heads)
        self._Collapse = Linear([heads, d_model//heads], d_model)


    def forward(self, query, key, value):

        assert key.shape[-2] = value.shape[-2]
        assert query.shape[-2]*self.query_compression == key.shape[-2]*self.content_compression


        #Localize all entries, and create the dilation heads.
        query = query.transpose(-1, -2) #(batch, d_model, items)
        key = key.transpose(-1, -2)
        value = value.transpose(-1, -2)

        local_queries = Glimpses.dilocal(query, self.query_kernel, self.query_stride, self.dilation) #(batch, d_model, head, same_item, query_local)
        local_keys = Glimpses.dilocal(key, self.content_kernel, self.content_stride, self.dilation)#(batch, d_model, head, same_item, local)
        local_values = Glimpses.dilocal(value, self.content_kernel, self.content_stride, self.dilation) #(batch, d_model, head, same_item, local)

        #Perform the heading interprojections, respectinve the existing heads

        local_queries = local_queries.transpose(-4, -2).tranpose(-1, -2) #(batch, item, head, local, d_model)
        local_keys = local_keys.transpose(-4, -2).transpose(-1, -2) #(batch, item, head, local, d_model)
        local_values = local_values.transpose(-4, -2).transpose(-1, -2) #(batch, item, head, local, d_model)

        local_queries = self._Query(local_queries) #(batch, item, head, query_local, d_small)
        local_keys = self._Key(local_keys) #(batch, item, head, content_local, d_small)
        local_values = self._Value(local_values) #(batch, item, head, content_local, d_small)

        #Perform attention on the local axis, and inject positional score information per head.

        local_queries = local_queries.transpose(-4, -2) #(batch, head, item, query_local, d_small)
        local_keys = local_keys.transpose(-4, -2) #(batch, head, item, local, d_small)
        local_values = local_values.transpose(-4, -2)

        score = torch.matmul(local_queries, local_keys.transpose(-1, -2)) #(batch, head, item, query_local, local)
        score = torch.softmax(score, dim=-1)
        attention = torch.matmul(score, local_values) #(batch, head, item, query_local, d_small)

        #Delocalize, combine, and return

        attention = attention.transpose(-1, -2).flatten(-3, -2) #(batch, head, item, d_small)
        attention = attention.transpose(-2, -1) #(batch, item, head, d_small)
        final_result = self._Collapse(attention) #(batch, item, d_model)
        return final_result








