"""

Submodels lie here


"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from .submodel import AbstractSubmodel


class LayerFactory():
    def __init__(self, layer: nn.Module,**kwargs):
        self.layer_factory = layer
        self.parameters = kwargs
    def __call__(self, **kwargs):
        call_kwargs = self.parameters.copy()
        for key, value in kwargs.items():
            call_kwargs[key] = value
        return self.layer_factory(**call_kwargs)


class AbstractSubmodelFactory():
    def __init__(self):
    def __call__(self, **kwargs)-> AbstractSubmodel:
        raise NotImplementedError("Must impliment call for an abstract factory")

class EmbeddingsSubmodelFactory(AbstractSubmodelFactory)
    def __init__(self, submodel_factory, layer_factories: List[LayerFactory]):
        super().__init__()
        self.submodel_factory = submodel_factory
        self.layers = layer_factories
    def __call__(self, **kwargs):


class CrossEntropyBoost(nn.Module):
    """
    Cross entropy with a bit of a twist.

    Each ensemble channel is independently
    processed, and then starting from channel one
    the results are merged. In particular, each
    channel contributes an additive factor to the
    final logits; additionally, each intermediate
    logit is evaluated and the bit losses are used
    to update the cross entropy weights.

    The net result is that if there was a lot of
    loss on the prior layer, this layer will aggressively train to
    minimize further loss.

    """
    def __init__(self,
                 logit_width: int,
                 ensemble_channel: int = 1,
                 label_smoothing: float = 0.0,
                 boost_smoothing: float = 0.3):

        super().__init__()
        self.logit_width = logit_width
        self.channel = ensemble_channel
        self.smoothing = label_smoothing
        self.boost = boost_smoothing
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, data: torch.Tensor, labels: torch.Tensor):

        #Prep and smooth labels.
        if labels.dtype == torch.int32 or labels.dtype == torch.int64 or labels.dtype == torch.int16:
            labels = F.one_hot(labels, self.logit_width).type(torch.float32)
        labels = (1-self.smoothing)*labels + self.smoothing/self.logit_width

        #Prep various channels
        channels = data.unbind(dim=self.channel)
        logit_shape = list(channels[0].shape[:-1]) + [self.logit_width]
        weights = torch.ones(logit_shape, dtype=data.dtype)
        logits = torch.zeros(logit_shape, dtype=data.dtype)
        loss = torch.tensor([0.0], dtype=data.dtype)

        #Run
        for channel in channels:

            #Update loss
            logits = logits + channel
            entropy = - labels * torch.log_softmax(logits, dim=-1)
            weighted_entropy = entropy*weights
            loss = loss + weighted_entropy.sum()/self.logit_width

            #Update and smooth weights.

            weights = entropy*(1-self.boost) + self.boost/self.logit_width
        return loss

