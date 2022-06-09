"""

Storing tensors such that torchscript can work with them can be
quite a pain. This set of tools makes it a lot easier. Tensors
are stored by placing them in the initialization region, and become
something that can then be accessed by looking at
.stored

"""

from __future__ import annotations


from typing import List, Optional, Union, Tuple, Dict

import torch
from torch import nn
from Utility.Torch.Models.SupertransformerOld import StreamTools
from Utility.Torch.Models.SupertransformerOld.StreamTools import StreamTensor

class TensorStorageItem(nn.Module):
    def __init__(self, tensor: torch.Tensor, requires_grad=False):
        super().__init__()
        self.item = nn.Parameter(tensor, requires_grad=requires_grad)
    def forward(self):
        return self.item


def DictTensorStorage(tensors: Dict[str, torch.Tensor], requires_grad=False):

    #Store
    storage = nn.ModuleDict()
    for name in tensors:
        storage[name] = TensorStorageItem(tensors[name], requires_grad)
    return storage
