# perform imports
from typing import Union, Sequence, Optional, Callable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import math
from functools import wraps

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

class BandedMultiheadedAttention(nn.Module):
    """
    This layer performs banded attention in a memory efficient manner in
    pytorch

    Banded matrix multiplication only evaluates values within a particular band
    of the braader matrix multiplication kernel. This has the advantage of significantly
    reducing the computation required on long sequences. Dilation, additionally, allowed
    the addressing of remote values using the same infrastructure. Both of these were
    used together to develop the LongFormer model for processing lang lengths of text,
    a recent development in the field of NLP.

    This is an implimentation and extension of this concept.

    --- details ---

    Attention is performed in distinct bands, reducing the amount of computation required to a much smaller
    banded kernel. Within this band, the banded attention is calculated as within LongFormer, but additional steps
    occur as well.

    #Local Kerneling and dilation

    A local kernel is generated to perform the attention earlier. The items in the kernel represent the
    relative position of items around the index of interest. Multiple indices can be handled by multiple kernels.
    In theory, each query from the input sees neighboring, proportional items from the keys and values. Additionally,
    setting the dilations allows for increasing the distance between entries of the keys and values.

    #Relative Position Supersampling

    One dilation head may contain information pertaining to many different relavent local phenomenom. This is the
    idea behind supersampling.

    At the Score step of attention, after multiplying the keys with the queries, but before the softmax, the result
    is expanded into multiple new heads, and goes through an additional linear layer, allowing the encoding of relative
    positional information into the result. This is the idea behind so called "supersampling"

    For each dilation, one can set a supersampling rate. This will increase the different positonal samples which
    can be chosen, and some experimentation may be needed to find the best rate.

    #Compression

    It is possible for a normal transformer to take arguments of variable length. This recovers some of this
    functionality.

    One can set a compression ratio, in terms of first query size, then content size, as in for instance (1, 1). This
    would represent the ratio 1:1. This is the standard setup, but it is not the only one.

    Lets say you have a query of dims (..., 100, 512) and keys, values of (..., 200, 512). In this case, there
    are two keys, values for every one query. One can then set the compression ratio to be (1: 2), and the
    class will ensure that keys are fed in at the appropriate rate.

    --- methods ---

    minimum_query_length: Returns what the minimal query length is. Pad to at least this long
    minimum_value_length: Returns what the minimal value length is. Pad to at least this long
    minimal_key_length: Returns the the minimal key length is. Pad to at least this long.

    """

    def minimum_query_length(self):
        return self.query_kernel
    def minimum_value_length(self):
        return self.content_kernel
    def minimum_key_length(self):
        return self.content_kernel
    def __init__(self,
                 d_model: int,
                 kernel_width: int,
                 d_internal: Optional[int] = None,
                 supersampling: Optional[List[int]] = None,
                 dilation_rates: Optional[List[int]] = None,
                 compression_ratio: Optional[Tuple[int, int]] = None,
                 pad: Optional[bool] = None,
                 trim: Optional[bool] = None
                 ):
        """

        :param d_model: The width of the embeddings
        :param kernel_width: How wide to make the base kernel. May in fact be enlarged, depending on the compression ratio
        :param dilation_rates: The dilation rate per subhead. Must match length of supersampling
        :param supersampling: How many supersamples to take per dilation head. Must match length of dilation_rates
        :param compression_ratio: The expected ratio of items in query to items in key, value. Must be given as
            Tuple[query, content] if defined. Is set at (1, 1) if not defined.
        :param pad: Whether or not to pad when the input query, key, or value is not long enough
        :param trim: Whether to trim off the extra padding if we did end up padding the input
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
        if pad is None:
            pad = False
        if trim is None:
            trim = False

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
        assert d_model > 0


        subheads = dilation_rates.shape[0]
        heads = supersampling.sum()
        if d_internal is None:
            d_internal = d_model//subheads

        Query_Projector = Linear(d_model, d_internal, subheads)
        Key_Projector = Linear(d_model, d_internal, subheads)
        Value_Projector = Linear(d_model, d_internal, heads)
        Collapse_Projector = Linear([heads, d_internal], d_model)
        Pos_Sampling = Linear([query_kernel, content_kernel], [query_kernel, content_kernel], heads)

        #Store

        self.dilation = dilation_rates
        self.sampling = supersampling
        self.heads = heads
        self.subheads = subheads
        self.d_model = d_model
        self.d_internal = d_internal
        self.compression = compression_ratio

        self.base_kernel = kernel_width

        self.query_kernel = query_kernel
        self.content_kernel = content_kernel

        self.query_stride = query_step
        self.content_stride = content_step

        self.run_pad = pad
        self.trim = trim

        self._Query = Query_Projector
        self._Key = Key_Projector
        self._Value = Value_Projector
        self._Sampler = Pos_Sampling
        self._Collapse = Collapse_Projector
    def pad(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            fill: float = 0.0):
        """

        Ensures the query and content contain the appropriate values
        to run attention on. Pads the query, key, and value to the correct lengths if not.
        Padding is inserted at the end.

        :param query: The query to pad
        :param key: The key to pad
        :param value: The value to pad.
        :return: padded_query, padded_key, padded_value
        """


        if query.shape[-2] < self.query_kernel:
            difference = int(self.query_kernel - query.shape[-2])
            query = F.pad(query, [0, 0, 0, difference], value=fill)
        if value.shape[-2] < self.content_kernel:
            difference = int(self.content_kernel - value.shape[-2])
            key = F.pad(key, [0, 0, 0, difference], value=fill)
            value = F.pad(value, [0, 0, 0, difference], value=fill)
        if query.shape[-2] % self.compression[0] != 0:
            difference = int(self.compression[0] - query.shape[-2] % self.compression[0])
            query = F.pad(query, [0, 0, 0, difference], value=fill)
        if value.shape[-2] % self.compression[1] != 0:
            difference = int(self.compression[1] - query.shape[-2] % self.compression[1])
            key = F.pad(key, [0, 0, 0, difference], value=fill)
            value = F.pad(value, [0, 0, 0, difference], value=fill)
        return query, key, value
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor):
        """


        :param query: Entry in (..., query_item, d_model) format, matching ratio
        :param key: Entry in (..., content_item, d_model) format, matching ratio
        :param value: Entry in (..., content_item, d_model) format, matching ratio
        :return:
        """

        assert isinstance(query, torch.Tensor)
        assert isinstance(key, torch.Tensor)
        assert isinstance(value, torch.Tensor)

        assert query.shape[-1] == key.shape[-1]
        assert query.shape[-1] == value.shape[-1]
        assert query.shape[-1] == self.d_model
        assert key.shape[-2] == value.shape[-2]

        query_kernel = self.query_kernel
        content_kernel = self.content_kernel

        if self.run_pad:
            revised_query, revised_key, revised_value = self.pad(query, key, value)
        else:
            if query.shape[-2] < self.query_kernel:
                raise ValueError("Item dimension of query is smaller then query kernel")
            if value.shape[-2] < self.content_kernel:
                raise ValueError("Item dimension of value is smaller then content kernel")
            if query.shape[-2]*self.compression[1] != value.shape[-2]*self.compression[0]:
                raise ValueError("Query and content lengths were not provided in the correct ratio")

            revised_query = query
            revised_key = key
            revised_value = value

        #Localize all entries, and create the dilation heads.
        revised_query = revised_query.transpose(-1, -2) #(..., d_model, items)
        revised_key = revised_key.transpose(-1, -2)
        revised_value = revised_value.transpose(-1, -2)

        local_queries = Glimpses.dilocal(revised_query, self.query_kernel.item(), self.query_stride.item(), self.dilation) #(batch, d_model, head, same_item, query_local)
        local_keys = Glimpses.dilocal(revised_key, self.content_kernel.item(), self.content_stride.item(), self.dilation)#(batch, d_model, head, same_item, local)
        local_values = Glimpses.dilocal(revised_value, self.content_kernel.item(), self.content_stride.item(), self.dilation) #(batch, d_model, head, same_item, local)
        local_values = torch.repeat_interleave(local_values, self.sampling, dim=-3)

        #Perform the heading interprojections for scoring

        local_queries = local_queries.transpose(-3, -2).transpose(-4, -1) #(batch, query_local, item, head, d_model)
        local_keys = local_keys.transpose(-3, -2).transpose(-4, -1) #(batch, local, item, head, d_model)
        local_values = local_values.transpose(-3, -2).transpose(-4, -1) #(..., item, head, local, d_model)

        local_queries = self._Query(local_queries) #(batch, query_local, item, head, d_small)
        local_keys = self._Key(local_keys)
        local_values = self._Value(local_values)

        #Perform attention on the local axis.

        local_queries = local_queries.transpose(-4, -2).transpose(-4, -3) #(batch, item, head, query_local, d_small)
        local_keys = local_keys.transpose(-4, -2).transpose(-4, -3) #(batch, item,  head, local, d_small)
        local_values = local_values.transpose(-4, -2).transpose(-4, -3)

        score = torch.matmul(local_queries, local_keys.transpose(-1, -2)) #(batch, item, head, query_local, local)
        score = torch.repeat_interleave(score, self.sampling, dim=-3) #Expand for supersampling
        score = self._Sampler(score) #Perform supersampling. Knowledge of relative order injected.
        score = torch.softmax(score, dim=-1)

        attention = torch.matmul(score, local_values) #(batch, item, head, query_local, d_small)

        #Delocalize, combine, trim and return

        attention = attention.transpose(-4, -3).flatten(-3, -2) #(batch, head, item, d_small)
        attention = attention.transpose(-3, -2) #(batch, item, head, d_small)
        final_result = self._Collapse(attention) #(batch, item, d_model)
        if self.trim and final_result.shape[-2] > query.shape[-2]:
            final_result = final_result[..., :query.shape[-2], :]
        return final_result

class FeedForward(nn.Module):
    """
    The transformer feedforward layer.

    Nothing much to see here.
    """

    def __init__(self, d_model, dim_feedforward=2048):
        super().__init__()
        self._ff1 = Linear(d_model, dim_feedforward)
        self._ff2 = Linear(dim_feedforward, d_model)
        self._activation = torch.relu
    def forward(self, tensor):
        tensor = self._ff1(tensor)
        tensor = self._activation(tensor)
        tensor = self._ff2(tensor)
        return tensor














