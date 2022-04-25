"""

The padding module contains functions which can be
utilized to perform torch based padding behavior
with relative ease.

Importantly, it also contains functions
which syncronized well with other functions.

"""
from typing import Sequence, Union, Optional

import torch
from torch import nn
from torch.nn import functional as F

def PadToCentered(tensor: torch.Tensor, length: int, dimension: int, fill=0.0)-> torch.Tensor:
    """

    Pads a tensor along the indicated dimension to a particular length, embedding
    the initial tensor in the center.

    :param tensor: the tensor to pad
    :param length: How long the tensor will need to be
    :param dimension: the dimension to pad
    :param fill: What to fill
    :return: A torch tensor
    """

    #Calculate the offset. Do this by calculating the midpoint, then subtracting off half
    #of the tensor's length

    midpoint = length//2
    offset = midpoint - (tensor.shape[dimension] + 1)//2

    #Execute pad, and return

    return PadToLength(tensor, length, dimension, offset, fill)


def PadToLength(tensor: torch.Tensor, length: int, dimension: int, offset: int, fill=0.0)-> torch.Tensor:
    """

    Pad a tensor along the indicated dimension with the indicated fill to the indicated length.
    Insert a section of zeros of length "offset" at the beginning of the tensor, and then
    pad everything else at the end.

    :param length: How long we must pad to
    :param dimension: The dimension to pad
    :param offset: How much to pad at the beginning
    :param fill: What the padding value will be
    :return: A padded tensor
    """
    #Isolate the modified dimension.
    tensor = tensor.transpose(-1, dimension)

    #Create the pad_op for functional padding. Also, assert it is the case that enough values exist
    outstanding = tensor.shape[0]-length
    assert outstanding >= 0, "Cannot pad a tensor to a shorter length"

    prior_padding = offset
    outstanding -= prior_padding
    assert outstanding >= 0, ("Offset is too large for this length", offset, length, outstanding)

    post_padding = outstanding
    pad_op = (prior_padding, post_padding)

    #Perform the padding. Return the result

    output = F.pad(tensor, pad_op, value=fill)
    return output.transpose(-1, dimension)




