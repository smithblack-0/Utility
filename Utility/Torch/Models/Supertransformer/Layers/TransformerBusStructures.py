"""

Structures designed to exist at the ensemble layer are located here. All
layers returned by these functions are in lists, for easy

"""
from typing import List, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers
from Utility.Torch.Models.Supertransformer.Layers.TransformerBusTools import\
    DataLayer, TaskMemoryLayer,  Data2TaskMemory, TaskMemory2Data

class TaskBusUnit(nn.Module):
    """

    An outer class, designed to allow the smooth flow of large quantities of
    information between some sort of large main bus and various subentries.

    Uses a LSTM pattern driven by the taskmemory bus, and only travels horizontally. The
    memory bus controls the LSTM like intake, and the hidden state is provided by the
    last submodel.

    Importantly, can be much, much, much wider if desired.
    """

class TaskProcessingUnit(nn.Module):
    """
    A reasonably adaptable task processing unit. Useful as a submodule top layer.

    Contains interchange actions, processing actions,
    and more. Must be provided with two internal processing
    units to function,

    Internal flow according to:
    #TODO: Update flowchart
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
        def forward(self, tensor):
            return self._norm(tensor +self._layer(tensor))
    def __init__(self,
                 d_data: int,
                 d_taskmemory: int,
                 data_processor: nn.Module,
                 taskmemory_processor: nn.Module,
                 transfer_heads: int,
                 dropout: float = 0.0):
        """
        :param d_data: How wide the data embeddings are
        :param d_taskmemory: How wide the task memory embeddings are
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
        #Attach layernorms to processing layers
        data_processor = self.AddLayerNorm(d_data, data_processor)
        taskmemory_processor = self.AddLayerNorm(d_taskmemory, taskmemory_processor)

        #Construct sequence to process results

        layers = nn.Sequential()
        layers.append(Data2TaskMemory(d_data, d_taskmemory, transfer_heads, dropout)) #Intake transfer
        layers.append(DataLayer(data_processor)) #Data processing
        layers.append(TaskMemoryLayer(taskmemory_processor)) #Task memory processing
        layers.append(TaskMemory2Data(d_data, d_taskmemory, transfer_heads, dropout)) #Task memory retransfer
        self._operation = layers
    def forward(self, data, taskmemory):
        return self._operation(data, taskmemory)

