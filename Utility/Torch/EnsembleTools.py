"""

Submodels lie here


"""
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from .Learnables import Layers

class AbstractSubmodel(nn.Module):
    """
    The abstract submodel. Defines the contract
    that must be followed for submodels to work.

    Responsible for the residual movement and the residual merge.
    """
    def __init__(self):
        super(AbstractSubmodel, self).__init__()
    def forward(self, input_tensor: torch.Tensor, residual: List[Optional[torch.Tensor]])\
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError("Forward must be implimented")

class AddMergeSubmodel(AbstractSubmodel):
    """
    The basic addmerge submodel. Delays layernorm until start.

    """

    def __init__(self,
                 embedding_width: int,
                 dropout: float,
                 layers: List[nn.Module],):
        super().__init__()
        norms = [nn.LayerNorm(embedding_width) for _ in layers]
        self.norms = nn.ModuleList(norms)
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_tensor: torch.Tensor, residual: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if residual is None:
            residual = [None]*len(self.layers)
        new_residuals: List[torch.Tensor] = []
        tensor = input_tensor
        for layer, residual, norm in zip(self.layers, residual, self.norm):
            if residual is not None:
                tensor = tensor + residual
            tensor = norm(tensor)
            tensor = self.dropout(layer(tensor)) + tensor
            new_residuals.append(tensor)
        return tensor, new_residuals

class ConcatMergeSubmodel(AbstractSubmodel):
    """
    Performs a concatenation of the residual entries,
    then merges the result together.

    """
    def __init__(self, embedding_width: int, dropout: float, layers: List[nn.Module]):
        super().__init__()
        norms = [nn.LayerNorm(embedding_width) for _ in layers]
        mergers = [nn.Linear(2*embedding_width, embedding_width) for _ in layers]

        self.norms = nn.ModuleList(norms)
        self.mergers = nn.ModuleList(mergers)
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_tensor: torch.Tensor, residuals: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if residuals is None:
            residuals = [None]*len(self.layers)

        new_residuals: List[torch.Tensor] = []
        tensor = input_tensor
        for layer, residual, norm, merger, in zip(self.layers, residuals, self.norm, self.merger):
            if residual is None:
                residual = torch.zeros_like(tensor)
            tensor = torch.concat([tensor, residual], dim=-1)
            tensor = merger(tensor)
            tensor = norm(tensor)
            tensor = self.dropoout(layer(tensor)) + tensor
            new_residuals.append(tensor)
        return tensor, new_residuals

class GateMergeSubmodel(AbstractSubmodel):
    """
    Merge residuals into stream using a logic
    gate. Specifically GRU.
    """
    def __init__(self, embedding_width: int, dropout: float, layers: List[nn.Module]):
        super().__init__()

        norms = [nn.LayerNorm(embedding_width) for _ in layers]
        gates = [nn.GRUCell(embedding_width, embedding_width) for _ in layers]

        self.norms = nn.ModuleList(norms)
        self.gates = nn.ModuleList(gates)
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_tensor: torch.Tensor, residuals: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if residuals is None:
            residuals = [None]*len(self.layers)
        new_residuals: List[torch.Tensor] = []
        tensor = input_tensor
        for layer, residual, norm, gate in zip(self.layers, residuals, self.norms, self.gates):
            if residual is None:
                residual = torch.zeros_like(tensor)
            shape = list(tensor.shape[:-1])
            tensor = gate(tensor.flatten(0, -2), residual.flatten(0, -2))
            tensor = tensor.unflatten(dim=0, unflattened_size= shape + [tensor.shape[-1]])
            tensor = norm(tensor)
            tensor = self.dropout(layer(tensor)) + tensor
            new_residuals.append(tensor)
        return tensor, new_residuals

class SuperEnsemble(nn.Module):
    def __init__(self, submodels: List[AbstractSubmodel], ensemble_channel=-3):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        self.channel = ensemble_channel
    def forward(self, tensor: torch.Tensor):
        channels = tensor.unbind(dim=self.channel)
        residuals = None
        outputs: List[torch.Tensor] = []
        for channel, submodel in zip(channels, self.submodels):
            output, residuals = submodel(channel, residuals)
            outputs.append(output)
        return torch.stack(outputs, dim=self.channel)

class CrossEntropyBoost(nn.Module):
    """
    Cross entropy with a bit of a twist.

    """
    def __init__(self,
                 embedding_width: int,
                 logit_width: int,
                 channels: int,
                 ensemble_channel: int = -3,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.logit_width = logit_width
        self.channel = ensemble_channel
        self.logit_projector = [nn.Linear(embedding_width, logit_width) for _ in range(channels)]
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, data: torch.Tensor, labels: torch.Tensor):
        if labels.dtype == torch.int32 or labels.dtype == torch.int64 or labels.dtype == torch.int16:
            labels = F.one_hot(labels, self.logit_width)
        channels = data.unbind(dim=self.channel)
        weights = torch.zeros_like(channels[0])
        logits = torch.zeros_like(channels[0])
        for projector, channel in zip(self.logit_projector, channels):
            logits = logits + projector(channel)
            if weights is not None:
                weights = weights*labels*torch.log_softmax(channel, dim=-1)
            else:
                weights = labels*torch.log_softmax(data, dim=-1)
            loss = loss + weights.sum()
            weights = torch.exp(weights)
        return loss

