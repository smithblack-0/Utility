from typing import List, Optional, Tuple, Dict

from torch import nn
import treetensor.torch as torch

from Utility.Torch.Models.Supertransformer import StreamTools


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