from typing import List, Optional, Tuple

import torch
from torch import nn


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


class AdditiveSubmodel(AbstractSubmodel):
    """
    The basic addmerge submodel. Delays layernorm until start.

    https://docs.google.com/drawings/d/1eW3DdGc2n1j0m_lT2fUjaS7fCtxErOJNiGnUPVhlHu4/edit

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
        self.length = len(layers)
    def forward(self, input_tensor: torch.Tensor, residuals: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        new_residuals: List[torch.Tensor] = []
        tensor = input_tensor
        iterator = zip(self.layers, self.norms)
        for i, (layer, norm) in enumerate(iterator):
            if residuals is not None:
                tensor = tensor + residuals[i]
            tensor = norm(tensor)
            tensor = self.dropout(layer(tensor)) + tensor
            new_residuals.append(tensor)
        return tensor, new_residuals


class ConcativeSubmodel(AbstractSubmodel):
    """
    Performs a concatenation of the residual entries,
    then merges the result together.

    https://docs.google.com/drawings/d/1DgBFOSDj8C_FyUozFQvNdFJL5IPc7SqL0sNUdCSO1aE/edit
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
        new_residuals: List[torch.Tensor] = []
        tensor = input_tensor
        iterator = zip(self.layers, self.norms, self.mergers)
        for i, (layer, norm, merger) in enumerate(iterator):
            if residuals is None:
                residual = torch.zeros_like(tensor)
            else:
                residual = residuals[i]
            tensor = torch.concat([tensor, residual], dim=-1)
            tensor = merger(tensor)
            tensor = norm(tensor)
            tensor = self.dropout(layer(tensor)) + tensor
            new_residuals.append(tensor)
        return tensor, new_residuals


class GateSubmodel(AbstractSubmodel):
    """
    Merge residuals into stream using a logic
    gate. Specifically GRU.

    https://docs.google.com/drawings/d/18KNfrRn1WTLPn1zEW-gCtP8MLtQQfQuNXsSmch2Zu38/edit
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
        new_residuals: List[torch.Tensor] = []
        tensor = input_tensor
        iterator = zip(self.layers, self.norms, self.gates)
        for i, (layer, norm, gate) in enumerate(iterator):
            if residuals is None:
                residual = torch.zeros_like(tensor)
            else:
                residual = residuals[i]
            shape = tensor.shape[:-1]
            tensor = gate(tensor.flatten(0, -2), residual.flatten(0, -2))
            tensor = tensor.unflatten(0, shape)
            tensor = norm(tensor)
            tensor = self.dropout(layer(tensor)) + tensor
            new_residuals.append(tensor)
        return tensor, new_residuals