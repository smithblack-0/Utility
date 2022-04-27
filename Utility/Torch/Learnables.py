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
        :param kernel_width: How wide to make the base kernel
        :param heads: How many different heads to make
        :param dilation_rates: The dilation rate per head. MUST match head length if defined
        :param The expected ratio of items in query to items in key, value. Must be given as
            Tuple[query, content] if defined. Is set at (1, 1) if not defined.
        """

        #Start torch
        super().__init__()

        #Perform a little verification and set defaults

        assert isinstance(d_model, int)
        assert isinstance(kernel_width, int)
        assert isinstance(heads, int)

        assert d_model >= 1
        assert kernel_width >= 1
        assert heads >= 1

        if dilation_rates is None:
            dilation_rates = [1]*heads
        if compression_ratio is None:
            compression_ratio = (1, 1)


        #Simplify the ratio down to it's smallest terms

        query_kernel_multiplier, content_kernel_multiplier = compression_ratio

        assert query_kernel_multiplier >= 1
        assert content_kernel_multiplier >= 1


        gcd = math.gcd(query_kernel_multiplier, content_kernel_multiplier)
        query_kernel_multiplier /= gcd
        content_kernel_multiplier /= gcd

        #handle dilation

        assert len(dilation_rates) == heads
        dilation_rates = torch.Tensor(dilation_rates)

        #Store persistant useful constants

        self.heads = heads
        self.query_kernel = query_kernel_multiplier*kernel_width
        self.content_kernel = content_kernel_multiplier*kernel_width
        self.base_kernel = kernel_width
        self.dilation = dilation_rates

        # Create projection layers.

        self._Query = Linear(d_model, d_model//heads, heads)
        self._Key = Linear(d_model, d_model//heads, heads)
        self._Value = Linear(d_model, d_model//heads, heads)
        self._Pos = Linear([self.content_kernel, d_model//heads], [self.content_kernel, d_model//heads], heads)
        self._Collapse = Linear([heads, d_model//heads], d_model)



    def forward(self, query, key, value):

        #Handle the cases where the query and content do not have the same item length, by
        #resizing stride and kernels as appropriate.
        query_length = query.shape[-2]
        content_length = key.shape[-2]
        kernel = self.kernel
        if query_length // content_length > 1:
            # Content fits into query more than once. Adjust step

            query_step = query_length // content_length
            content_step = 1
        elif content_length // query_length > 1:
            #Query fits into content more than once. Adjust step

            query_step = 1
            content_step = content_length // query_length
        else:
            query_step = 1
            content_step = 1

        query_kernel = query_step
        content_kernel = content_step * kernel

        if query_length > content_length:
            query_kernel = query_kernel + query_length % content_length
        if content_length > query_length:
            content_kernel = content_kernel + content_length % query_length

        #Localize all entries, and create the dilation heads.
        query = query.transpose(-1, -2) #(batch, d_model, items)
        key = key.transpose(-1, -2)
        value = value.transpose(-1, -2)

        local_queries = Glimpses.dilocal(query, query_kernel, query_step, self.dilation) #(batch, d_model, head, same_item, query_local)
        local_keys = Glimpses.dilocal(key, content_kernel, content_step, self.dilation)#(batch, d_model, head, same_item, local)
        local_values = Glimpses.dilocal(value, content_kernel, content_step, self.dilation) #(batch, d_model, head, same_item, local)

        #Perform the heading interprojections, respectinve the existing heads

        local_queries = local_queries.transpose(-3, -2).transpose(-4, -1) #(batch, query_local, item, head, d_model)
        local_keys = local_keys.transpose(-3, -2).transpose(-4, 1) #(batch, local, item, head, d_model)
        local_values = local_values.transpose(-3, -2).transpose(-4, -1)

        local_queries = self._Query(local_queries) #(batch, query_local, item, head, d_small)
        local_keys = self._Key(local_keys)
        local_values = self._Value(local_values)

        #Perform attention on the local axis

        local_queries = local_queries.transpose(-4, -2) #(batch, head, item, query_local, d_small)
        local_keys = local_keys.transpose(-4, -2) #(batch, head, item, local, d_small)
        local_values = local_values.transpose(-4, -2)

        score = torch.matmul(local_queries, local_keys.transpose(-1, -2)) #(batch, head, item, query_local, local)
        score = torch.softmax(score, dim=-1)
        attention = torch.matmul(score, local_values) #(batch, head, item, query_local, d_small)

        #Delocalize, combine, and return

        attention = attention.transpose(-1, -2).flatten(dim=-1) #(batch, head, item, d_small)
        attention = attention.transpose(-2, -1) #(batch, item, head, d_small)
        final_result = self._Collapse(attention) #(batch, item, d_model)
        return final_result








