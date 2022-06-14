from typing import List

import torch
from torch import nn

from Utility.Torch.Archive.EnsembleTools.submodel import AbstractSubmodel
        
class SuperEnsemble(nn.Module):
    def __init__(self, submodels: List[AbstractSubmodel], ensemble_channel=1):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        self.channel = ensemble_channel
    def resolve(self, tensor: torch.Tensor):
        assert len(self.submodels) == tensor.shape[self.channel]
        channels = tensor.unbind(dim=self.channel)
        residuals = None
        outputs: List[torch.Tensor] = []
        for channel, submodel in zip(channels, self.submodels):
            output, residuals = submodel(channel, residuals)
            outputs.append(output)
        return torch.stack(outputs, dim=self.channel)

    def forward(self, tensor: torch.Tensor):
        pass