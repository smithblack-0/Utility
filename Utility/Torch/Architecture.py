"""

This module contains classes and functions which do not themselves contain
trainable parameters but instead can be used to prepare data for further processing.

"""
from typing import List

import torch
from torch import nn
import numpy as np

from Utility.Torch import Glimpses

class GLASuperprocesser(nn.Module):
    """

    Global Local Attention Superprocesser
    Reduces into ratios.
    Processes by items in module.
    Recombines.

    """

    def __init__(self,
                 d_models: List[int],
                 ratios: List[int],
                 modules: List[nn.Module]):
        """
        :param d_models: The width of the reserved embeddings in the incoming tensors
        :param ratios: The ratios of the reduced tensor sections to each other. Note that uniquely,
         zero will specify these are global embeddings. Note also that it MUST be the case that
         the tensor item dimension is a multiple of lcm(ratios), excluding zeros.
        :param modules: Modules to associate with, and process, each substream. Optional
        """
        super().__init__()

        self._d_models = d_models
        self._ratios = ratios
        self._modules = modules

    def forward(self, tensor: torch.Tensor):

        #Basic sanity checking
        assert torch.is_tensor(tensor)
        assert tensor.dim() >= 2
        assert tensor.shape[-1] == sum(self._d_models)
        assert tensor.shape[-2] % np.lcm.reduce(self._ratios) == 0


        #Functional defintion, with forking allowed.
        subsections = tensor.split(self._d_models, dim=-1)
        repetitions = [tensor.shape[-2]//item if item > 0 else tensor.shape[-2] for item in self._ratios]
        futures = []
        for item, ratio, module, repetitions in zip(subsections, self._ratios, self._modules, repetitions):
            #Define operations
            def setup(item: torch.Tensor):
                return item.transpose(-1, -2)
            def local(item: torch.Future):
                return Glimpses.local(item.value(), ratio, ratio, 1)
            def reduce(item: torch.Future):
                return item.value().max(dim=-1)
            def process(item: torch.Future):
                return module(item.value())
            def expand(item: torch.Future):
                return torch.repeat_interleave(item.value(), repetitions, dim=-1)
            def restore(item: torch.Future):
                return item.value().transpose(-1, -2)

            #Create chain
            future = torch.jit.fork(setup, item)
            future.then(local)
            future.then(reduce)
            future.then(process)
            future.then(expand)
            future.then(restore)

            #Append to futures
            futures.append(future)

        #Wait, concat the items together, then return.
        outcomes = torch.futures.wait_all(futures)
        output = torch.concat(outcomes, dim=-2)
        return output









