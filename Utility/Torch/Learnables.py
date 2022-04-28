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
        elif torch.is_tensor(head_shapes) and head_shapes.dim() == 0:
            head_shapes = [head_shapes]
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
                 supersampling: Optional[List[int]] = None,
                 dilation_rates: Optional[List[int]] = None,
                 compression_ratio: Optional[Tuple[int, int]] = None,
                 ):
        """

        :param d_model: The width of the embeddings
        :param kernel_width: How wide to make the base kernel
        :param dilation_rates: The dilation rate per subhead. Must match length of supersampling
        :param supersampling: How many supersamples to take per dilation head. Must match length of supersampling
        :param compression_ratio: The expected ratio of items in query to items in key, value. Must be given as
            Tuple[query, content] if defined. Is set at (1, 1) if not defined.
        """

        #Start torch
        super().__init__()

        #Defaults

        if supersampling is None:
            supersampling = [5, 5, 2, 1, 1]
        if dilation_rates is None:
            dilation_rates = [1, 1, 2, 4, 8]
        if compression_ratio is None:
            compression_ratio = (1, 1)

        #Perform a little verification

        assert isinstance(d_model, int)
        #Simplify the ratio down to it's smallest terms, and setup the kernel sizes

        assert isinstance(kernel_width, int)
        assert kernel_width >= 1
        assert isinstance(compression_ratio, (list, tuple))
        assert isinstance(compression_ratio[0], int)
        assert isinstance(compression_ratio[1], int)
        assert compression_ratio[0] >=1
        assert compression_ratio[1] >=1

        query_width = torch.tensor([1], dtype=torch.int64)
        kernel_width = torch.tensor(kernel_width, dtype=torch.int64)

        query_kernel_multiplier, content_kernel_multiplier = compression_ratio
        gcd = math.gcd(query_kernel_multiplier, content_kernel_multiplier)
        query_kernel_multiplier = query_kernel_multiplier//gcd
        content_kernel_multiplier = content_kernel_multiplier // gcd

        query_kernel = query_width*query_kernel_multiplier
        content_kernel = kernel_width*content_kernel_multiplier

        query_step = torch.tensor(query_kernel_multiplier, dtype=torch.int64)
        content_step = torch.tensor(content_kernel_multiplier, dtype=torch.int64)

        #Verify the dilation rates, sampling rates, and setup the dilation headspace

        assert isinstance(dilation_rates, (list, tuple))
        assert isinstance(supersampling, list)
        assert len(dilation_rates) == len(supersampling)
        for dilation, sample in zip(dilation_rates, supersampling):
            assert isinstance(dilation, int)
            assert dilation >= 1

            assert isinstance(sample, int)
            assert sample >= 0

        supersampling = torch.tensor(supersampling, dtype=torch.int64)
        dilation_rates = torch.tensor(dilation_rates, dtype=torch.int64)
        #Create projection parameters, and projectors

        assert isinstance(d_model, int)

        subheads = dilation_rates.shape[0]
        heads = supersampling.sum()
        d_headed = torch.floor_divide(d_model, subheads).type(torch.int64)
        assert d_headed >= 1

        Query_Projector = Linear(d_model, d_headed, subheads)
        Key_Projector = Linear(d_model, d_headed, subheads)
        Value_Projector = Linear(d_model, d_headed, heads)
        Collapse_Projector = Linear([heads, d_headed], d_model)
        Pos_Sampling = Linear([query_kernel, content_kernel], [query_kernel, content_kernel], heads)



        #Store

        self.dilation = dilation_rates
        self.sampling = supersampling
        self.heads = heads
        self.subheads = subheads
        self.d_model = d_model
        self.d_headed = d_headed

        self.base_kernel = kernel_width

        self.query_kernel = query_kernel
        self.content_kernel = content_kernel

        self.query_stride = query_step
        self.content_stride = content_step

        self._Query = Query_Projector
        self._Key = Key_Projector
        self._Value = Value_Projector
        self._Sampler = Pos_Sampling
        self._Collapse = Collapse_Projector


    def forward(self, query, key, value):
        """


        :param query: Entry in (..., query_item, d_model) format, matching ratio
        :param key: Entry in (..., content_item, d_model) format, matching ratio
        :param value: Entry in (..., content_item, d_model) format, matching ratio
        :return:
        """

        assert torch.is_tensor(query)
        assert torch.is_tensor(key)
        assert torch.is_tensor(value)

        assert query.shape[-1] == key.shape[-1]
        assert query.shape[-1] == value.shape[-1]
        assert query.shape[-1] == self.d_model

        #Localize all entries, and create the dilation heads.
        query = query.transpose(-1, -2) #(..., d_model, items)
        key = key.transpose(-1, -2)
        value = value.transpose(-1, -2)

        local_queries = Glimpses.dilocal(query, self.query_kernel, self.query_stride.item(), self.dilation) #(batch, d_model, head, same_item, query_local)
        local_keys = Glimpses.dilocal(key, self.content_kernel, self.content_stride.item(), self.dilation)#(batch, d_model, head, same_item, local)
        local_values = Glimpses.dilocal(value, self.content_kernel, self.content_stride.item(), self.dilation) #(batch, d_model, head, same_item, local)
        local_values = torch.repeat_interleave(local_values, self.sampling, dim=-3)

        #Perform the heading interprojections for scoring

        local_queries = local_queries.transpose(-3, -2).transpose(-4, -1) #(batch, query_local, item, head, d_model)
        local_keys = local_keys.transpose(-3, -2).transpose(-4, -1) #(batch, local, item, head, d_model)
        local_values = local_values.transpose(-3, -2).transpose(-4, -1)

        local_queries = self._Query(local_queries) #(batch, query_local, item, head, d_small)
        local_keys = self._Key(local_keys)
        local_values = self._Value(local_values)

        #Perform attention on the local axis

        local_queries = local_queries.transpose(-4, -2).transpose(-4, -3) #(batch, item, head, query_local, d_small)
        local_keys = local_keys.transpose(-4, -2).transpose(-4, -3) #(batch, item,  head, local, d_small)
        local_values = local_values.transpose(-4, -2).transpose(-4, -3)

        score = torch.matmul(local_queries, local_keys.transpose(-1, -2)) #(batch, item, head, query_local, local)
        score = torch.repeat_interleave(score, self.sampling, dim=-3)
        score = self._Sampler(score)
        score = torch.softmax(score, dim=-1)

        attention = torch.matmul(score, local_values) #(batch, item, head, query_local, d_small)

        #Delocalize, combine, and return

        attention = attention.transpose(-4, -3).flatten(-3, -2) #(batch, head, item, d_small)
        attention = attention.transpose(-3, -2) #(batch, item, head, d_small)
        final_result = self._Collapse(attention) #(batch, item, d_model)
        return final_result








