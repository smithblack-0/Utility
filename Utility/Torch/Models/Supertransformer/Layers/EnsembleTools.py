from __future__ import annotations


import copy
import math
from typing import List, Optional, Union, Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers
from Utility.Torch.Models.Supertransformer.Layers import StreamTools
from Utility.Torch.Models.Supertransformer.Layers import StorageTools
from Utility.Torch.Models.Supertransformer.Layers.StreamTools import StreamTensor

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

class tensorstorage(nn.Module):
    """
    Tensor storage as a module
    """
    def __getitem__(self, item):
        return getattr(self, item)
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        super().__init__()
        for name, tensor in tensors:
            assert hasattr(self, name) is False
            self.register_buffer(name, tensor)


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
                residuals_stream: List[StreamTools.StreamTensor],
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
                residuals_stream: List[StreamTools.StreamTensor],
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None,
                )\
            -> Tuple[StreamTools.StreamTensor, List[StreamTools.StreamTensor]]:

        new_residuals: List[StreamTools.StreamTensor] = []
        stream = input_stream
        for i, layer in enumerate(self._layers):
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
                residuals_stream: List[StreamTools.StreamTensor] = None,
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
            residual = residuals_stream[i]
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
    individual subinstance. This is abstract, and merely
    defines the correct implimentation.
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                input_stream: StreamTools.StreamTensor,
                ensemble_stream: Optional[StreamTools.StreamTensor] = None,
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None)\
        -> StreamTools.StreamTensor:
        raise NotImplementedError("Must impliment forward in EnsembleStartup")

class ConcatStartup(AbstractEnsembleStartup):
    """
    A class to build a functioning startup sequence based
    on the assumption that we should be building
    by concatenation. Contains a long term memory
    seed which is concatenated to the input each time
    """
    def __init__(self, mem_seed: StreamTensor):
        super().__init__()
        self.seeds = StorageTools.DictTensorStorage(mem_seed.stream)
        self._cache = StreamTensor
    def forward(self,
                input_stream: StreamTools.StreamTensor,
                ensemble_stream: Optional[StreamTools.StreamTensor] = None,
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None) -> StreamTools.StreamTensor:
        seeds = dict(self.seeds)
        seeds = {name: value() for name, value in seeds.items()}
        seeds = StreamTensor(seeds)

        if ensemble_stream is None:
            ensemble_stream = StreamTensor()

        merger = StreamTools.StreamMerger([input_stream, ensemble_stream, seeds])
        merger.stream.concat(dim=-1)
        merger.losses.sum()
        output_stream = merger.build()

        return output_stream

class ReductionStartup(AbstractEnsembleStartup):
    """
    A startup class, this is responsible for
    starting the direct tensor flow through
    the ensemble. This flavor performs
    reductive combination, adding tensors of the same
    shape together.
    """
    def __init__(self, mem_seed: StreamTensor, mode: str):
        super().__init__()
        self.seeds = StorageTools.DictTensorStorage(mem_seed.stream)
        self._cache = StreamTensor
        self._mode = mode

    def forward(self,
                input_stream: StreamTools.StreamTensor,
                ensemble_stream: Optional[StreamTools.StreamTensor] = None,
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None) -> StreamTools.StreamTensor:
        seeds = dict(self.seeds)
        seeds = {name: value() for name, value in seeds.items()}
        seeds = StreamTensor(seeds)

        if ensemble_stream is None:
            ensemble_stream = StreamTensor()

        merger = StreamTools.StreamMerger([input_stream, ensemble_stream, seeds])
        merger.stream.reduce_mode(self._mode)
        merger.losses.sum()
        output_stream = merger.build()

        return output_stream

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
            -> Tuple[StreamTensor, StreamTensor]:
        """

        :param ensemble_stream: The current stream from the current instance
        :param cumulative_stream: The cumulative stream. From the previous iteration.
        :param auxiliary_stream: Anything from the auxiliary stream
        :return: Two items. First, the cumulative stream tensor. Second, the task streamtensor.
        """
        raise NotImplementedError("Must impliment forward function in EnsembleTeardown")

class AbstractResStartup(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,
                res_stream: Optional[List[StreamTensor]],
                auxiliary_stream: Optional[List[StreamTensor]]
                ) -> List[StreamTensor]:
        raise NotImplementedError()

class


class SuperEnsemble(nn.Module):
    """

    A SuperEnsemble model. Consists of a
    collection of residually active submodels, and memory starts.
    May generate only it's own memory starts, or accept augmentive
    information from prior models as well.
    """

    def __init__(self,
                 Start: List[AbstractEnsembleStartup],
                 ResStart: List[AbstractResStartup],
                 TearDown: List[AbstractEnsembleTeardown],
                 SubModels: List[AbstractSubModel],
                 ):


        super().__init__()

        self.starters = Start
        self.res = ResStart
        self.teardown = TearDown
        self.submodels = SubModels

    def forward(self,
                input_stream: StreamTensor,
                residuals_stream: Optional[List[StreamTensor]],
                ensemble_streams: Optional[List[StreamTensor]],
                auxiliary_stream: Optional[StreamTensor])\
            -> Tuple[StreamTensor, List[StreamTensor], List[StreamTensor]]:

        """

        :param input_stream: The input to the model
        :param residuals_stream: Any prior residuals generated to incorporate
        :param ensemble_stream: Any prior per task information or conditioning to incoporate.
        :param auxiliary_stream: A place to put information that will be fed to each submodel.
        :return: A stream tensor, the output. A List of StreamTensors,
            one per submodel task recursive information. A list of StreamTensors, consisting
            of residual information.
        """
        #Strip out stream losses, metrics. Put them in null stream, to merge into the output later.
        null_stream = input_stream.keeponly([]) #Only keeps metrics, losses.
        input_stream = input_stream.branch(input_stream.names)

        #Preprocess, start tensors for ensemble
        if ensemble_streams is not None:
            iterate = zip(ensemble_streams, self.starters)
            stream = [start(input_stream, ensemble_stream, auxiliary_stream) for ensemble_stream, start in iterate]
        else:
            stream = [start(input_stream, None, auxiliary_stream) for start in self.starters]

        if residuals_stream is not None:
            iterate = zip(residuals_stream, self.res)
            residuals = [res_start(residual, auxiliary_stream) for residual, res_start in iterate]
        else:
            residuals = [res_start(None, auxiliary_stream) for res_start in self.res]

        #Perform submodel ensemble processing
        outputs: List[StreamTensor] = []
        for substream, submodel in zip(stream, self.submodels):
            output, residuals = submodel(substream, auxiliary_stream)
            outputs.append(output)

        #collapse the accumulated ensemble
        ensemble_items: List[StreamTensor] = []
        cumulative: Optional[StreamTensor] = None
        for output, final in zip(outputs, self.teardown):
            cumulative, ensemble_item = final(output, cumulative, auxiliary_stream)
            ensemble_items.append(ensemble_item)

        #Integrate metrics, losses back into stream.
        merging = StreamTools.StreamMerger([cumulative, null_stream])
        merging.losses.sum()
        merging.stream.sum()
        output = merging.build()

        #Return
        return output, residuals, ensemble_items







