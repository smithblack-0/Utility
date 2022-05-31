from collections import namedtuple
from typing import Optional, Tuple, NamedTuple, List

import torch
from torch import nn

from Utility.Torch.Models.Supertransformer import StreamTools, StorageTools
from Utility.Torch.Models.Supertransformer.StreamTools import StreamTensor

"""

This module is responsible for holding
ensemble Startup units and their requirements. 

A startup unit takes four parameters, and uses it to construct
the correct submodel stream and recursive stream. 

The four inputs they must handle are:

input_stream: Fresh data we are attempting to fit. Per batch.
ensemble_stream: An item from a previous superensemble we are continuing. Per batch.
recursive_stream: A stream generated from the corrolated teardown on a prior batch. Things placed
    in the recursive stream are generally expected to persist.
auxiliary_stream: The same throughout the model. A good place to put globally relavent things,
    like training mode and label. Per batch.

The output is two streamtensors. The first one is the submodel stream, consisting of
batch specific information to feed to a submodel. The second one is the recursive
stream, consisting of things to keep around over many generations.

"""

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
                recursive_stream: Optional[StreamTools.StreamTensor] = None,
                auxiliary_stream: Optional[NamedTuple] = None)\
        -> StreamTensor:
        raise NotImplementedError("Must impliment forward in EnsembleStartup")



class ConcatStartup(AbstractEnsembleStartup):
    """
    A class to build a functioning startup sequence based
    on the assumption that we should be building
    by concatenation. Contains a long term memory
    seed which is concatenated to the input each time

    Recursion will not be operational on the first loop. As
    such, an additional feature known as "recursive defaults" exists
    to allow padding to be placed on a tensor when the recursive
    input is not yet fully developed.
    """
    def __init__(self,
                 mem_seed: Optional[StreamTensor] = None,
                 recursive_defaults: Optional[StreamTensor] = None):

        super().__init__()
        if mem_seed is None:
            mem_seed = StreamTensor()

        self.seeds = StorageTools.DictTensorStorage(mem_seed.stream)
        self.defaults = StorageTools.DictTensorStorage(recursive_defaults.stream)
    def forward(self,
                input_stream: Optional[StreamTensor] = None,
                ensemble_stream: Optional[StreamTensor] = None,
                recursive_stream: Optional[StreamTensor] = None,
                auxiliary_stream: Optional[StreamTensor] = None) -> StreamTensor:

        seeds = {name: value().clone() for name, value in self.seeds.items()}
        seeds = StreamTensor(seeds)

        if input_stream is None:
            input_stream = StreamTensor()
        if ensemble_stream is None:
            ensemble_stream = StreamTensor()
        for name in self.defaults():
            if name not in ensemble_stream:
                ensemble_stream = ensemble_stream.set[name, self.defaults[name]().clone()]
        if recursive_stream is None:
            defaults = {name: value().clone() for name, value in self.defaults.items()}
            recursive_stream = StreamTensor(defaults)

        merger = StreamTools.StreamMerger([input_stream, ensemble_stream, recursive_stream, seeds])
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
    def __init__(self, mem_seed: StreamTensor, mode: str = 'sum'):
        """

        :param mem_seed: The memory seeds. Consists of learnable tensors of some sort. These will
            automatically be broadcast across the input.
        :param mode: The reduction mode. Allowed options are "sum", "mean", "min", "max", "median", and "first"
        """
        super().__init__()
        self.seeds = StorageTools.DictTensorStorage(mem_seed.stream, requires_grad=True)
        self._mode = mode

    def forward(self,
                input_stream: Optional[StreamTools.StreamTensor] = None,
                ensemble_stream: Optional[StreamTools.StreamTensor] = None,
                recursive_stream: Optional[StreamTools.StreamTensor] = None,
                auxiliary_stream: Optional[NamedTuple] = None) ->\
            StreamTensor:

        seeds = {name: value().clone() for name, value in self.seeds.items()}
        seeds = StreamTensor(seeds)

        if input_stream is None:
            input_stream = StreamTensor()
        if ensemble_stream is None:
            ensemble_stream = StreamTensor()
        if recursive_stream is None:
            recursive_stream = StreamTensor()

        merger = StreamTools.StreamMerger([input_stream, ensemble_stream, recursive_stream, seeds])
        merger._stream.reduce_mode(self._mode)
        merger.losses.sum()
        output_stream = merger.build()

        return output_stream

class StartupCollection(nn.Module):
    """
    A location for a collection of startup actions to dwell within.
    """
    def __init__(self, startups: List[AbstractEnsembleStartup]):
        super().__init__()
        self.startups = nn.ModuleList[startups]
    def forward(self,
                input_stream: Optional[StreamTools.StreamTensor] = None,
                ensemble_streams: Optional[List[StreamTools.StreamTensor]] = None,
                recursive_streams: Optional[List[StreamTools.StreamTensor]] = None,
                auxiliary_stream: Optional[NamedTuple] = None) ->\
            List[StreamTensor]:

        auxiliary_dict = auxiliary_stream._asdict()
        if input_stream is None:
            input_stream = StreamTensor()
        if auxiliary_stream is None:
            auxiliary_class = namedtuple('auxiliary_stream', ['channel'])
        else:
            auxiliary_class = namedtuple('ensemble_stream', list(auxiliary_dict.keys()) + ['channel'])

        outputs: List[StreamTensor] = []
        for i, layer in self.startups:
            if ensemble_streams is None:
                ensemble = StreamTensor()
            else:
                ensemble = ensemble_streams[i]
            if recursive_streams is None:
                recusive = StreamTensor()
            else:
                recusive = recursive_streams[i]

            auxiliary_dict['channel'] = torch.tensor(i, dtype=torch.int32)
            auxiliary = auxiliary_class(**auxiliary_dict)
            output = layer(input_stream, ensemble, recusive, auxiliary)
            outputs.append(output)

        return outputs

