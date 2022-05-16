"""

Layers responsible for exchanging information between task and data streams
live here. Options include data to task only, task to data only, and
residual exchange prepwork.

"""

from typing import List, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers




class Data2TaskMemory(nn.Module):
    """
    Transfers information from the data
    stream to the task memory stream using a transformer.
    Includes residual passthrough and layernorm after transfer
    """
    def __init__(self,
                 d_data: int,
                 d_taskmemory: int,
                 heads: int,
                 dropout: float,
                 layernorm: bool = True,
                 bypass: bool = True):
        super().__init__()

        self._attn = nn.MultiheadAttention(d_taskmemory, heads, dropout, kdim=d_data, vdim=d_data)
        self._layernorm = layernorm
        self._bypass = bypass
        if layernorm:
            self._norm = nn.LayerNorm(d_taskmemory)
    def forward(self, data, taskmemory):

        update = self._attn(taskmemory, data, data)
        if self._bypass:
            update = update + taskmemory
        if self._layernorm:
            update = self._norm(update)
        return data, update

class TaskMemory2Data(nn.Module):
    """
    Transfers information from the taskmemory
    stream into the data stream using a transformer layer.
    """

    def __init__(self,
                 d_data: int,
                 d_memory: int,
                 heads: int,
                 dropout: float,
                 layernorm: bool = True,
                 bypass: bool = True):
        super().__init__()

        self._attn = nn.MultiheadAttention(d_data, heads, dropout, kdim=d_memory, vdim=d_memory)
        self._layernorm = layernorm
        self._bypass = bypass
        if layernorm:
            self._norm = nn.LayerNorm(d_data)
    def forward(self, data, taskmemory):
        update = self._attn(data, taskmemory, taskmemory)
        if self._bypass:
            update = update + data
        if self._layernorm:
            update = self._norm(update)
        return update, taskmemory
class DataLayer(nn.Module):
    """
    A layer which one wishes only to act on the data
    stream may be placed here. Includes a residual
    bypass and layernorm by default
    """
    def __init__(self,
                layer: nn.Module):
        """
        :param layer: The layer to apply
        """
        super().__init__()
        self._layer = layer
    def forward(self, data, taskmemory):
        data = self._layer(data)
        return data, taskmemory

class TaskMemoryLayer(nn.Module):
    """
    A layer which one wishes only to act on the taskmemory
    stream may be placed here. Includes a residual
    bypass and layernorm by default
    """

    def __init__(self,
                 layer: nn.Module,
                 ):
        """
        :param d_model: The enbedding width on layer EXIT
        :param layer: The layer to apply
        :param bypass: Whether to perform residual bypass
        :param layernorm: Whether to perform layernorm.
        """
        super().__init__()

        self._layer = layer

    def forward(self, data, taskmemory):
        taskmemory = self._layer(taskmemory)
        return data, taskmemory

