"""

Structures designed to exist at the ensemble layer are located here. All
layers returned by these functions are in lists, for easy

"""
from typing import List, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers

class TaskProcessingUnit(nn.Module):
    """
    A reasonably adaptable task processing unit. Useful
    as a submodel layer. Consists of data, bus channels
    which are residually interchanged between model
    and submodel sections.

    Requires a data processing and taskmemory processing
    stack to be provided upon creation.

    Internal flow according to:
    https://docs.google.com/drawings/d/1Ej0ZlPbTqyDC_aC1xiMwghGn28IDDC8667dgyHr-I0Y/edit
    """
    class AddLayerNorm(nn.Module):
        """
        A small helper class for layernorming
        """
        def __init__(self, d_model: int, layer: nn.Module):
            super().__init__()
            self._layer = layer
            self._norm = nn.LayerNorm(d_model)
        def forward(self, tensor, *args):
            return self._norm(tensor +self._layer(tensor, *args))
    def __init__(self,
                 d_data: int,
                 d_bus: int,
                 d_taskmemory: int,
                 transfer_heads: int,

                 data_processor: nn.Module,
                 taskmemory_processor: nn.Module,

                 dropout: float = 0.0):
        """
        :param d_data: How wide the data embeddings are
        :param d_bus: How wide the bus embeddings are
        :param d_taskmemory: How wide the task memory embeddings should be. Should be smaller than
            d_bus, generally.
        :param data_processor: The data processor stack. Usually a
            transformer encoder stack or derivative
        :param taskmemory_processor:
            The memory processor stack. Usually a transformer encoder
            or derivative
        :param transfer_heads:
            The width of the heads used in the transfer process
        :param dropout:
            The strength of the transfer dropout.
        """
        super().__init__()

        #Create intake norms

        intake_data = nn.LayerNorm(d_data)
        intake_bus = nn.LayerNorm(d_bus)

        #Create bus to taskmemory bottleneck

        bottleneck = nn.Sequential()
        bottleneck.append(nn.Dropout(dropout))
        bottleneck.append(nn.Linear(d_bus, d_data))
        bottleneck.append(nn.ReLU())

        #Create add+layernorm for processing stacks

        data_processor = self.AddLayerNorm(d_data, data_processor)
        taskmemory_processor = self.AddLayerNorm(d_taskmemory, taskmemory_processor)

        #Create data to taskmemory transfer, taskmemory to data transfer

        data2memory = nn.MultiheadAttention(d_taskmemory, transfer_heads, dropout, kdim=d_data, vdim=d_data, batch_first=True)
        data2memory = self.AddLayerNorm(d_taskmemory, data2memory)

        memory2data = nn.MultiheadAttention(d_data, transfer_heads, dropout, kdim=d_taskmemory, vdim=d_taskmemory, batch_first=True)
        memory2data = self.AddLayerNorm(d_data, memory2data)

        #Create bus rebuild

        debottleneck = nn.Linear(d_taskmemory, d_bus)

        #Store layers

        self._intake_data = intake_data
        self._intake_bus = intake_bus

        self._bottleneck = bottleneck
        self._debottleneck = debottleneck

        self._memory2data = memory2data
        self._data2memory = data2memory

        self._datastack = data_processor
        self._memorystack = taskmemory_processor

    def forward(self, data, bus):

        #Layernorms, including residuals from other submodels
        data = self._intake_data(data)
        bus = self._intake_bus(bus)

        #Convert to task memory, bus to task collapse, then transfer memory
        taskmemory = self._bottleneck(bus)
        taskmemory = self._data2memory(taskmemory, data, data)

        #Processing

        taskmemory = self._memorystack(taskmemory)
        data = self._datastack(data)

        #Memory to data, debottlenecking, and final processing

        data = self._memory2data(data, taskmemory, taskmemory)
        bus = self._debottleneck(bus) + bus

        return data, bus



