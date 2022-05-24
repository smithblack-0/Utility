from typing import Optional

from torch import nn

from Utility.Torch.Models.Supertransformer import StreamTools, StorageTools
from Utility.Torch.Models.Supertransformer.StreamTools import StreamTensor


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