from __future__ import annotations
from typing import List, Optional, Union, Tuple, Dict, NamedTuple
from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers
from Utility.Torch.Models.SupertransformerOld import StreamTools


def _L1(tensor: torch.Tensor):
    """
    Calculates the L1 loss
    """
    tensor = torch.abs(tensor)
    tensor = tensor.mean()
    return tensor


def _L2(tensor: torch.Tensor):
    """
    Calculates the L2 Loss
    """
    tensor = tensor ** 2
    tensor = tensor.mean()
    return tensor


class BypassCore(nn.Module):
    """
    A layer which contains within it the core shared parameters, including
    weights, biases, and loss calculations.
    """

    loss_maps = {'L1': _L1, 'L2': _L2}
    gain_loss_name = 'BypassGainLoss'
    cross_loss_name = 'BypassCrossLoss'

    @property
    def weight(self) -> torch.Tensor:
        return self._weight

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def bottleneck_dim(self) -> int:
        return self._bottleneck_width

    def __init__(self,
                 input_dim: int,
                 bottleneck_dim: int,
                 bypass_dim: int,
                 loss_type: str = 'L2',
                 device=None,
                 dtype=None
                 ):
        super().__init__()

        assert loss_type in self.loss_maps
        internal_dim = bottleneck_dim + bypass_dim
        weight = torch.empty([internal_dim, input_dim], device=device, dtype=dtype)
        weight = nn.init.orthogonal_(weight)
        weight = nn.Parameter(weight)

        self._bottleneck_width = bottleneck_dim
        self._input_dim = input_dim

        self._weight = weight
        self._loss = loss_type

    def loss(self):
        loss_matrix = torch.matmul(self.weight, self.weight.transpose(-1, -2))
        identity = torch.eye(loss_matrix.shape[0], dtype=self.weight.dtype, device=self.weight.device)

        loss_matrix = loss_matrix - identity
        gain_loss_logits = torch.diagonal(loss_matrix)
        cross_loss_logits = torch.flatten(loss_matrix.masked_fill(identity == 1, 0))

        gain_loss = self.loss_maps[self.loss](gain_loss_logits)
        cross_loss = self.loss_maps[self.loss](cross_loss_logits)

        return {self.gain_loss_name : gain_loss, self.cross_loss_name : cross_loss}


class BypassStart(nn.Module):
    """
    A layer which is utilizable for starting a bypass. Takes a
    StreamTensor, and targets a particular sub-tensor. This sub-tensor
    is projected using the core bypass information, then the source
    tensor goes away and the projected names are inserted into the stream.
    """

    def __init__(self,
                 intake_name: str,
                 bottleneck_name: str,
                 bypass_name: str,
                 parameter_module: BypassCore):
        super().__init__()

        self.intake_name = intake_name
        self.bottleneck_name = bottleneck_name
        self.bypass_name = bypass_name

        self.core = parameter_module

    def forward(self, stream: StreamTools.StreamTensor):
        # Check exists
        assert self.intake_name in stream.names

        # Perform bypass setup projection.

        (tensor) = stream.isolate([self.intake_name])

        tensor = tensor.unsqueeze(-1)
        tensor = torch.matmul(self.core.weight, tensor)
        tensor = tensor.squeeze(-1)

        # Split up tensors. Make stream
        bottleneck = tensor[..., :self.core.bottleneck_dim]
        bypass = tensor[..., self.core.bottleneck_dim:]

        # Join bypass. Return stream

        new_stream_content = StreamTools.StreamTensor({"bottleneck" : bottleneck, "bypass" : bypass})
        stream = StreamTools.stream_merge([stream, new_stream_content])
        return stream



class BypassStop(nn.Module):
    """
    This layer takes from a stream two related tensors, and
    puts them back together to form a whole. It also
    calculates any needed losses, and tracks them
    """

    def __init__(self,
                 bottleneck_name: str,
                 bypass_name: str,
                 output_name: str,
                 parameters: BypassCore
                 ):
        super().__init__()

        self.output_name = output_name
        self.bottleneck_name = bottleneck_name
        self.bypass_name = bypass_name

        self.core = parameters

    def forward(self, stream: StreamTools.StreamTensor):

        # Check sanity. Combine tensors
        assert self.bottleneck_name in stream.names
        assert self.bypass_name in stream.names

        bottleneck, bypass = stream.isolate([self.bottleneck_name, self.bypass_name])
        tensor = torch.concat([bottleneck, bypass], dim = -1)

        # Perform restorative projection.
        tensor = tensor.unsqueeze(-1)
        tensor = torch.matmul(self.core.weight.transpose(-1, -2), tensor)
        tensor = tensor.squeeze(-1)



        #Create stream update. Merge. Finalize
        stream_items = {self.output_name : tensor}
        losses = self.core.loss()

        stream_update = StreamTools.StreamTensor(stream_items, losses, None)
        stream = StreamTools.stream_merge([stream, stream_update])
        return stream
