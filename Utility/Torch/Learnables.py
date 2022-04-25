# perform imports
from typing import Union, Sequence, Optional, Callable, List

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
    @staticmethod
    def localize(tensor, kernel, head, striding, dilation):
        tensor = tensor[..., head, :]
        tensor = Glimpses.local(tensor, kernel, striding, dilation)
        return tensor
    @staticmethod
    def band_attn(result):
        """ Performs attention """

        query, key, value = result.value()
        query = query.value() #(..., band, d_query)
        key = key.value() #(..., band, d_content)
        value = value.value() #(..., items, band, d_content)

        query = query
        key = key.transpose(-1, -2)
        value = value

        score = query.matmul(key)
        score = torch.softmax(score, dim=-1)

        outcome = score.matmul(value).sum(dim=)
        return outcome

    def _uneven_length_compensation(self, query_length, content_length):
        """

        :param kernel: The length of the kernel
        :param query_length: The length of the query
        :param content_length: The length of the content
        :return:
        """

        # Create the initial kernel shapes. Adapt for cases
        # where one kernel is several factors larger than another
        # by adjusting the kernel size and step rate
        kernel = self.kernel_shape
        if query_length // content_length > 1:
            # These are different by enough to adjust the step factor

            query_step = query_length // content_length
            content_step = 1
        elif content_length // query_length > 1:

            query_step = 1
            content_step = content_length // query_length
        else:
            query_step = 1
            content_step = 1

        query_kernel = query_step * kernel
        content_kernel = content_step * kernel

        if query_length > content_length:
            query_kernel = query_kernel + query_length % content_length
        if content_length > query_length:
            content_kernel = content_kernel + content_length % query_length

        return (query_kernel, query_step),  (content_kernel, content_step)
    def __init__(self,
                 d_query: int,
                 d_content: int,
                 kernel_width: int,
                 heads: int = None,

                 query_dilations: Optional[Sequence[int]] = None,
                 content_dilations: Optional[Sequence[int]] = None,
                 ):
        #Start torch
        super().__init__()

        #assert

        assert isinstance(d_query, int)
        assert isinstance(d_content, int)
        assert isinstance(kernel_width, int)
        assert isinstance(heads, int)

        make = lambda x: torch.Tensor(x)
        query_dilations = make(query_dilations) if query_dilations is not None else torch.ones([heads])
        content_dilations = make(content_dilations) if content_dilations is not None else torch.ones([heads])

        assert query_dilations.dim() == 1
        assert content_dilations.dim() == 1

        assert heads == query_dilations.shape[0]
        assert heads == content_dilations.shape[0]

        #Passed. Create basic parameter arrays.

        self._Query = Linear(d_query, (heads, d_query//heads))
        self._Key = Linear(d_content, (heads, d_content//heads))
        self._Value = Linear(d_content, (heads, d_content//heads))
        self._Collapse = Linear((heads, d_content//heads), d_content)

        #Store constants

        self.kernel_shape = kernel_width
        self.query_dilations = query_dilations
        self.content_dilations = content_dilations
        self.heads = heads
    def forward(self, query, key, value):

        #Perform the heading projections

        query = self._Query(query)
        key = self._Key(key)
        value = self._Value(value)

        #As it stands at this point, it may be the case that the query and content length do not
        #match. To compensate for this, develop a kernel-striding combination by which the quantities
        #in each nicely line up. After this, the total number of bands will be the same between the
        #sections

        query_length = query.shape[-2]
        content_length = key.shape[-2]
        compensation = self._uneven_length_compensation(query_length, content_length)
        (query_kernel, query_stride), (content_kernel, content_stride) = compensation

        #Working with different dilations is not very parallizable natively. Use jit futures to allow
        #forking

        localization_futures = []
        for head, query_dil, content_dil in zip(range(self.heads), self.query_dilations, self.content_dilations):
            local_futures = []
            local_futures.append(torch.jit.fork(self.localize, query, query_kernel,
                                               head, query_stride, query_dil))
            local_futures.append(torch.jit.fork(self.localize, key, content_kernel,
                                               head, content_stride, content_dil))
            local_futures.append(torch.jit.fork(self.localize, value, content_kernel,
                                               head, content_stride, content_dil))
            total_future = torch.futures.collect_all(local_futures)
            localization_futures.append(total_future)

        attention_futures = []
        for item in localization_futures:
            attention_futures.append(item.then(self.attn))

        attention_results = torch.futures.wait_all(attention_futures)
        attention_results = torch.stack(attention_results, dim=-2)

        final_result = self._Collapse(attention_results)








