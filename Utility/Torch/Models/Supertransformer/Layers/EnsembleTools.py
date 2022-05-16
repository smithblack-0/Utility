from __future__ import annotations


import copy
import math
from typing import List, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers


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

def make_SuperEnsemble(total_submodels: int,
                       memory_shape: List[int],
                       base_model: List[nn.Module],
                       memory_dtype: torch.dtype = torch.float32,
                       copy_layers: bool = True,
                       reinitialize_layers: bool = True) -> SuperEnsemble:
    """

    :param total_submodels: How many submodels to create.
    :param memory_shape: The shape of the memory starts.
    :param base_model: A list of layers, representing the sequential layers to be called in
        the model.
    :param memory_dtype: The dtype of the memory. Should be same as data stream
    :param copy_layers: Whether or not to create indepedent layers for each submodel
    :param reinitialize_layers: If making independent layers, whether to attempt to reinitialize
        each layers parameter. Will look for, and call, reset_parameters where it can find it
    :return: A new SuperEnsemble model
    """

    #This loop creates the copies of the submodel, and memory starts, needed
    #to start the SuperEnsemble.
    submodels = []
    memstarts = []
    for _ in range(total_submodels):
        #Handle submodel layer creation
        if copy_layers is False:
            submodel = base_model
        else:
            submodel = copy.deepcopy(base_model)

            if reinitialize_layers is True:
                #Reinitialize all parameters that are possible, recursively
                for sublayer in submodel:
                    for microlayer in sublayer.modules():
                        if hasattr(microlayer,  'reset_parameters'):
                            microlayer.reset_parameters()

        #Handle submodel creation, memstart creation, and storage.
        submodel = SubModel(submodel)
        memstart = MemSeed(memory_shape, memory_dtype)
        submodels.append(submodel)
        memstarts.append(memstart)

    #With everything created, we create the superensemble and return

    return SuperEnsemble(submodels, memstarts)





class SubModel(nn.Module):
    """
    A single submodel. This is responsible for taking
    in a memory start, a tensor start, and the current memory
    plus tensor residuals. It then performs processing based on
    this, creating outputs for all of these. It is a single
    sequential submodel in a model super ensemble.

    It must display the requested memory start
    as well.
    """
    def __init__(self,
                sublayers: List[nn.Module],
                 ):
        super().__init__()
        layers = [torch.jit.script(layer) for layer in sublayers]
        self._layers = nn.ModuleList(layers)
    def forward(self,
                tensor: torch.Tensor,
                memory: torch.Tensor,
                tensor_residuals: Optional[List[torch.Tensor]] = None,
                memory_residuals: Optional[List[torch.Tensor]] = None,
                ):

        new_tensor_residuals = []
        new_memory_residuals = []
        tensor = tensor
        for i, layer in enumerate(self._layers):

            #Get tensor residuals. Handle cases where nonexistant.
            if tensor_residuals is None:
                tensor_residual = torch.zeros_like(tensor)
            else:
                tensor_residual = tensor_residuals[i]

            #Get memory residuals
            if memory_residuals is None:
                memory_residual = torch.zeros_like(memory)
            else:
                memory_residual = memory_residuals[i]

            #Update, calculate, apply

            tensor = tensor + tensor_residual
            memory = memory + memory_residual
            tensor, memory = layer(tensor, memory)

            new_tensor_residuals.append(tensor)
            new_memory_residuals.append(memory)
        return tensor, memory, new_tensor_residuals, new_memory_residuals



class MemSeed(nn.Module):
    """
    A small seed, required to start the memory process rolling. One
     must be provided for every submodel. These are trainable.
    """

    def __init__(self,
                 shape: List[int], dtype: torch.dtype = torch.float32):
        super().__init__()
        seed = torch.zeros(shape, dtype=dtype, requires_grad=True)
        unsqueeze_times = 0

        while seed.dim() < 2:
            #edge case for insufficient dims
            seed = seed.unsqueeze(0)
            unsqueeze_times += 1

        torch.nn.init.kaiming_uniform_(seed, math.sqrt(5))

        for _ in range(unsqueeze_times):
            #Return to normal.
            seed = seed.squeeze(0)

        self._seed = nn.Parameter(seed)
    def forward(self):
        return self._seed




class SuperEnsemble(nn.Module):
    """

    A SuperEnsemble model. Consists of a
    collection of residually active submodels, and memory starts.
    May generate only it's own memory starts, or accept augmentive
    information from prior models as well.
    """

    def __init__(self,
                 stream_submodels: List[nn.Module],
                 task_seeds: List[MemSeed],
                 ):
        """
        :param task_seeds: A list of memory task seeds, one per submodel
        :param stream_submodels: A list of submodels.
        """

        super().__init__()

        self._memory_seeds = nn.ModuleList(task_seeds)
        self._submodels = nn.ModuleList(stream_submodels)

    def forward(self,
                data_stream: torch.Tensor,
                tensor_residuals: Optional[List[torch.Tensor]] = None,
                task_residuals: Optional[List[torch.Tensor]] = None,
                task_augments: Optional[List[torch.Tensor]] = None) \
            -> Tuple[Tuple[List[torch.Tensor], List[torch.Tensor]], Tuple[List[torch.Tensor], List[torch.Tensor]]]:

        """


        :param data_stream: The incoming primary data to process
        :param task_residuals: Any prior residuals along the memory stream, presumably task related
        :param tensor_residuals: Any prior tensor residuals to include
        :param task_augments: Anything to inject right after the memory seeds.
        :return:
        """
        tensor_residuals = tensor_residuals
        memory_residuals = task_residuals
        tensor_outputs: List[torch.Tensor] = []
        memory_outputs: List[torch.Tensor] = []

        counter = 0
        for layer, seed in zip(self._submodels, self._memory_seeds):
            #Fetch memory. Augment seed.
            memory = seed()

            if task_augments is not None:
                #Insert the augment stream if available.
                memory = memory + task_augments[counter]
                counter += 1

            outcome = layer(data_stream, memory, tensor_residuals, memory_residuals)
            tensor, memory, tensor_residuals, memory_residuals = outcome
            tensor_outputs.append(tensor)
            memory_outputs.append(memory)
        return (tensor_outputs, memory_outputs), (tensor_residuals, memory_residuals)

