"""

A holder for a variety of functions each of which
somehow rearrange the parts of a tensor without
particularly modifying the underlying values

(sans padding)




"""

import torch
from torch import nn
from torch.nn import functional as F

import numbers
import numpy as np


def view(tensor, input_shape: "tuple, int", output_shape: "tuple, int"):
    """
    Description:

    An improved form of pytorch's view. This will, when passed an input shape and output shape,
    fill in any missing dimensions from tensor's shape argument, then attempt to turn the last
    dimensions from input shape to output shape. It also accepts implicit commands,

    For instance, if input shape was (5, 2), and output was 10, the section of shape
    5,2 would be reshaped to match the section of shape 10.





    """

    # Convertion

    if isinstance(input_shape, numbers.Number):
        input_shape = [input_shape]
    if isinstance(output_shape, numbers.Number):
        output_shape = [output_shape]

    # Basic sanity testing
    assert np.prod(input_shape) == np.prod(output_shape), "Input shape and output shape were not compatible"

    slice_length = len(input_shape)
    assert np.array_equal(input_shape, tensor.shape[-slice_length:]), "Input shape and tensor shape not compatible"

    # Construct view resize

    new_view = [*tensor.shape[:-slice_length], *output_shape]

    # view. Return
    return tensor.view(new_view)


def local(tensor, kernel_width, stride_rate, dilation_rate, pad=False):
    """

        A function to produce local views of the last dimension of a tensor. These are
        views, indexed along the second to last dimension, with content along the last
        dimension, which are the precursors to convolution, possessing a kernel width,
        stride rate, and dilation rate as defined in the local view class.

        Enabling padding prevents information loss due to striding.
        """

    # Construct shape. Take into account the kernel_width, dilation rate, and stride rate.

    # The kernel width, and dilation rate, together modifies how far off the end of the
    # data buffer a naive implimentation would go, in an additive manner. Striding, meanwhile
    # is a multiplictive factor

    compensation = (kernel_width - 1) * dilation_rate  # calculate dilation-kernel correction
    final_index_shape = tensor.shape[-1] - compensation  # apply
    assert final_index_shape > 0, "Configuration is not possible - final kernel exceeds available tensors"
    final_index_shape = final_index_shape // stride_rate  # Perform striding correction.
    final_shape = (*tensor.shape[:-1], final_index_shape, kernel_width)  # Final shape

    # Construct the stride. The main worry here is to ensure that the dilation striding, and primary
    # striding, now occurs at the correct rate. This is done by taking the current one, multiplying,
    # and putting this in the appropriate location.

    final_stride = (*tensor.stride()[:-1], stride_rate * tensor.stride()[-1], dilation_rate * tensor.stride()[-1])

    # Pad tensor, if requested, to prevent loss of information. All padding goes on the end

    if pad:
        tensor = F.pad(tensor, (0, compensation))

        # Finish by striding and returning

    return tensor.as_strided(final_shape, final_stride)
