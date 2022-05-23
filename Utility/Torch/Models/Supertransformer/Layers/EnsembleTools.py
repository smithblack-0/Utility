from __future__ import annotations


import copy
import math
from typing import List, Optional, Union, Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers
from Utility.Torch.Models.Supertransformer.Layers import StreamTools


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

class AbstractSubModel(nn.Module):
    """
    A class defining the interface for residually
    connective submodels.

    --- forward method params ---

    input_stream: The constructed input from the appropriate EnsembleStartup. A StreamTensor
    residuals_stream: A collection of StreamTensors representing the residuals produced in prior layers.
    auxilary_stream: A signaling and utility stream. All layers have equal access to this.
    """
    def __init__(self):
        super().__init__()
    def forward(self,
                input_stream: StreamTools.StreamTensor,
                residuals_stream: Optional[List[StreamTools.StreamTensor]] = None,
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None)\
            -> Tuple[StreamTools.StreamTensor, List[StreamTools.StreamTensor]]:
        raise NotImplementedError("Must impliment forward in AbstractSubModel")

class ReducingSubModel(AbstractSubModel):
    """
    A single residually active submodel. Uses combinative reduction
    for residuals. May perform many residuals to one merge.
    """
    def __init__(self, sublayers: List[nn.Module]):


        super().__init__()
        layers = [torch.jit.script(layer) for layer in sublayers]
        self._layers = nn.ModuleList(layers)

    def forward(self,
                input_stream: StreamTools.StreamTensor,
                residuals_stream: Optional[List[StreamTools.StreamTensor]] = None,
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None,
                )\
            -> Tuple[StreamTools.StreamTensor, List[StreamTools.StreamTensor]]:

        new_residuals: List[StreamTools.StreamTensor] = []
        stream = input_stream
        for i, layer in enumerate(self._layers):
            if residuals_stream is not None:
                residual = residuals_stream[i]
                merger = StreamTools.StreamMerger([stream, residual])
                merger.stream.reduce_mode(self._mode)
                merger.losses.sum()
                stream = merger.build()
            stream = layer(stream)
            new_residuals.append(stream)
        final_stream = stream
        return final_stream, new_residuals

class ConcatSubModel(AbstractSubModel):
    """
     A single residually active submodel. Uses concatenation to manage
     it's residuals.

     Each iteration, the prior residuals are concatenated
     on, the layer is run, and then the appropriate units are split off.
     """
    def __init__(self,
                 sublayers: List[nn.Module],
                 defaults: Dict[str, int]):


        super().__init__()
        layers = [torch.jit.script(layer) for layer in sublayers]
        self._layers = nn.ModuleList(layers)
        self._defaults = defaults
        self._zeros = StreamTools.StreamTensor({key: [value] for (key, value) in defaults})
    def forward(self,
                input_stream: StreamTools.StreamTensor,
                residuals_stream: Optional[List[StreamTools.StreamTensor]] = None,
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None,
                )\
            -> Tuple[StreamTools.StreamTensor, List[StreamTools.StreamTensor]]:

        assert self._defaults.keys() in input_stream
        initial_widths = {name: input_stream.stream[name].shape[-1] for name in self._defaults.keys()}
        concat_widths = self._defaults
        breakup_directive = {name: (-1, [initial_widths[name], concat_widths[name]]) for name in initial_widths}

        new_residuals = []
        stream = input_stream
        for i, layer in enumerate(self._layers):
            if residuals_stream is not None:
                residual = residuals_stream[i]
            else:
                stream_data = {}
                for name, length in self._defaults:
                    shape = list(stream.stream[name].shape[:-1]) + [length]
                    tensor = torch.zeros(shape)
                    stream_data[name] = tensor
                residual = StreamTools.StreamTensor(stream_data)
            merger = StreamTools.StreamMerger([stream, residual])
            merger.stream.concat(dim=-1)
            merger.losses.sum()
            stream = merger.build()
            stream: StreamTools.StreamTensor = layer(stream)
            stream, residual = stream.split(breakup_directive) #Note stream must come first to avoid losing losses.
            new_residuals.append(residual)
        final_stream = stream
        return final_stream, new_residuals

class AbstractEnsembleStartup(nn.Module):
    """
    A class responsible for taking an incoming ensemble
    stream or group, and creating a new stream for the
    individual subinstance
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                input_stream: StreamTools.StreamTensor,
                recursive_stream: Optional[StreamTools.StreamTensor] = None,
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None)\
        -> StreamTools.StreamTensor:
        raise NotImplementedError("Must impliment forward in EnsembleStartup")

class AbstractEnsembleTeardown(nn.Module):
    """
    A class responsible for finishing up with
    an ensemble of some sort and narrowing the
    ensemble back down to a single output stream. This is a single
    subinstance.

    -- forward params --

    ensemble_stream: The item from the currently evaluated ensemble.
    cumulative_stream: The item from the last evaluated ensemble. Optional.
    auxiliary_stream: Any additional information the user may want, such as losses
        or training commands. Optional.
    """
    def __init__(self):
        super().__init__()
    def forward(self,
                ensemble_stream: StreamTools.StreamTensor,
                cumulative_stream: Optional[StreamTools.StreamTensor] = None,
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None) \
            -> StreamTools.StreamTensor:
        raise NotImplementedError("Must impliment forward function in EnsembleTeardown")

class SuperEnsemble(nn.Module):
    """
    A model consisting of a collection of submodels which are residually
    interconnected.
    """


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

