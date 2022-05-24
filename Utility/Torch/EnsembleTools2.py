from __future__ import annotations


import copy
import math
from typing import List, Optional, Union, Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers
from Utility.Torch.Models.Supertransformer import StreamTools
from Utility.Torch.Models.Supertransformer.StreamTools import StreamTensor

"""

The text flow encoding level. This is fairly comparible
to modern NPL machine learning algorithms, though with a 
little extra depth. 

DIAGRAMS:

https://docs.google.com/drawings/d/1Ej0ZlPbTqyDC_aC1xiMwghGn28IDDC8667dgyHr-I0Y/edit


TERMINOLOGY:


** Architecture **

Stream_Submodel: A single stream of predicitive information traveling from the source data, 
    accepting prior residuals, and returning processed information
FlowExchange: The process of exchanging information between the memory and stream tensors
within a submodel, while doing stream level processing in between.
ResidualBypass: The process of reinserting individual sublayer outputs into the next
sublayer input.


** Tensors **

TextStream: A text based embedding of some sort. Arbitrary length
Memory: A tensor of some sort. Known length. Attends to an entire text stream

FEATURES:

- Effective depth

First, many parallel stacks of the same sequence of model exists,
with residual redirect connecting them. Data may take as short,
 or as long, a path as it wishes before reaching the end,
 avoiding a large portion of the dead gradient problems by ensuring
 the distance between a calculation which has drawn a conclusion,
 and the exit point, is very short.

- Boosting and Pretraining

The existance of many parallel stacks allows the usage of a useful technique - boosting.
During pretraining, it is possible to place a projection stack on the end of each individual
submodel, and check how far off from true the particular example is. 

"""

class AbstractProcessingUnit(nn.Module):
    """
    A layer container which is capable of performing
    a single operation in an ensemble chain. Accepts a collection
    of inputs drawn from the indicated dependency tree.'

    It is highly recommended to make the dependencies direct. This is,
    however, not required.
    """
    @property
    def dependencies(self)-> List[Tuple[int, int]]:
        raise NotImplementedError()
    @property
    def defaults(self) -> List[Tuple[int, int]]:
        raise NotImplementedError()
    def __init__(self):
        super().__init__()
    def forward(self, input_streams: List[StreamTensor]):
        raise NotImplementedError()

class SuperEnsemble(nn.Module):
    """
    A class dedicated to running the superensemble
    process, by first compiling the dependencies
    then executing in the appropriate order.
    """
    class ProcessingNode(nn.Module):

        def __init__(self,

        def forward(self, rank: int, item: StreamTensor):








    def __init__(self,
                 ensemble: List[List[AbstractProcessingUnit]],
                 compile_loops: int = 100):

        super().__init__()

        #Basic sanity checking
        for sublist in ensemble:
            assert len(sublist) == len(ensemble[0])

        x_dim = len(ensemble)
        y_dim = len(ensemble[0])

        #Compile interrelations mask. This indicates, in the absolute frame,
        #what elements are needed for each particular unit to fire.

        interrelations = torch.full([x_dim, y_dim, x_dim, y_dim], False)
        for i, submodel in enumerate(ensemble):
            for j, unit in enumerate(submodel):
                dependencies = unit.dependencies
                for x, y in dependencies:
                    new_x = x + i
                    new_y = y + j
                    if  (x >= 0 and x < x_dim) and (y >= 0 and y < y_dim):
                        interrelations[i, j, new_x, new_y] = True

        #Figure out the firing order information.
        has_fired = torch.full([x_dim, y_dim], False)
        firing_iteration = torch.zeros([x_dim, y_dim], dtype=torch.int32)

        for iteration in range(compile_loops):
            for i in range(x_dim):
                for j in range(y_dim):
                    if not has_fired[i, j]:
                        mask = interrelations[i, j]
                        elements = has_fired.masked_select(mask)
                        if torch.all(elements):
                            #prerequisites met. Fire
                            firing_iteration[i, j] = iteration
                            has_fired[i,j] = True
            if torch.all(has_fired):
                break




