"""

Layers responsible for exchanging information between task and data streams
live here. Options include data to task only, task to data only, and
residual exchange prepwork.

"""
from __future__ import annotations
from typing import List, Optional, Union, Tuple, Dict, NamedTuple
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers


"""

As the complexity of the models I am working with has grown,
using tuples to track information is becoming increasingly
untenable. Any minor modification requires major changes to all layers.

As a result, I now utilize the stream mechanism. This involves placing
information into a dictionary, known as the stream, from which it can be
later retrieved. 

Each stream iteration should be unique, so as to allow functional
processing. 

"""


@torch.jit.script
class StreamTensor():
    """

    A stream consists of a key-value map with an arbitrary
    number of named tensors, and an entry called "losses"
    which is another dictionary."""

    @property
    def names(self):
        return self._stream.keys()
    @property
    def stream(self):
        return self._stream
    @property
    def losses(self):
        return self._losses
    @property
    def metrics(self):
        return self._metrics
    @property
    def residuals(self):
        return self._residuals

    def clone(self)-> StreamTensor:
        """
        Creates a structually independent clone of the stream

        :param stream: A dictionary in stream format
        :return: A clone of the dictionary, with the dictionary itself being independent, but
        the entries being the same. Modifying the data structure does not modify the original
        """
        return StreamTensor(self._stream, self._losses, self._metrics, self._residuals)

    def isolate(self, names: List[str]) -> List[torch.Tensor]:
        """
        Extracts the specified entries out of the stream in the order
        given in the list.

        :param names: A list of strings, referring to names to extract
        :exception: If one of the names was not part of the stream.
        :return: A list of the items, in the given order
        """
        output = []
        for name in names:
            assert name in self._stream, "name not found in stream"
            output.append(self._stream[name])
        return output

    #Stream modification functions
    def branch(self, names: List[str]) -> StreamTensor:
        """
        Isolates a substream out of the main stream. Discards any
        metric or loss data in the branch tensor

        :param names: The names to isolate
        :return: StreamTensor
        """
        new_stream = {}
        for name in names:
            assert name in self.stream
            new_stream[name] = self.stream[name]

        return StreamTensor(new_stream, None, None, None)


    def discard(self, names: List[str]) -> StreamTensor:
        """
        Discards from the stream the items with the indicated names

        :param names: The names to discard
        :return: A StreamTensor
        """
        new_stream = self.stream.copy()
        for name in names:
            assert name in new_stream
            new_stream.pop(name)

        return StreamTensor(new_stream, self.losses, self.metrics, self._residuals)

    def add_res(self, name: str,  residual: torch.Tensor)-> StreamTensor:
        """

        Adds a residual to the current main branch.

        :param name: The name it is found under
        :param residual: The residual
        :return: StreamTensor
        """

        residuals = self._residuals.copy()
        if name not in residuals:
            residuals[name] = []

        item = residuals[name].copy()
        item.append(residual)
        residuals[name] = item

        return StreamTensor(self.stream, self.losses, self.metrics, residuals)
    def __init__(self,
                 stream: Optional[Dict[str, torch.Tensor]] = None,
                 losses: Optional[Dict[str, torch.Tensor]] = None,
                 metrics: Optional[Dict[str, List[torch.Tensor]]] = None,
                 residuals: Optional[Dict[str, List[torch.Tensor]]] = None):

        if stream is None:
            stream = {}
        if losses is None:
            losses = {}
        if metrics is None:
            update: Dict[str, List[torch.Tensor]] = {}
        else:
            update = metrics
            torch.jit.annotate(Dict[str, List[torch.Tensor]], update)

        if residuals is None:
            update2: Dict[str, List[torch.Tensor]] = {}
        else:
            update2 = residuals
            torch.jit.annotate(Dict[str, List[torch.Tensor]], update2)

        self._stream = stream
        self._losses = losses
        self._metrics = update
        self._residuals = update2



@torch.jit.script
def reduce(reduce: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    """ Combines a list of tensors together, and then reduces it"""
    assert reduce in ("sum", "mean", "max", "min", "median"), "reduction method must be sum, mean, max, min, median, or first"
    stacked = torch.stack(tensors, dim=0)
    if reduce == "sum":
        return torch.sum(stacked, dim=0)
    if reduce == "mean":
        return torch.mean(stacked, dim=0)
    if reduce == "max":
        maximums, _ = torch.max(stacked, dim=0)
        return maximums
    if reduce == "min":
        minimums, _ = torch.min(stacked, dim=0)
        return minimums
    if reduce == "median":
        medians, _ = torch.median(stacked, dim=0)
        return medians
    if reduce == "first":
        return stacked[0]
    raise RuntimeError("Illegal state")

@torch.jit.script
def get_raw_merge(streams: List[StreamTensor]) -> \
        Tuple[
            Dict[str, List[torch.Tensor]],
            Dict[str, List[torch.Tensor]],
            Dict[str, List[torch.Tensor]]]:
    """

    Merges together the streams to yield dictionaries of
    stream, losses, and metrics. The losses and stream
    dictionary must be further reduced to eliminate duplicate
    entries.

    The results for stream, losses may be though of a
    key to list mapping, where each list contains all the
    instances of that key that popped up in any of the streams.
    """
    # Create merge builders.
    new_stream: Dict[str, List[torch.Tensor]] = {}
    new_losses: Dict[str, List[torch.Tensor]] = {}
    new_metric: Dict[str, List[torch.Tensor]] = {}
    for stream in streams:

        # Transfer stream into builder
        for name in stream.stream:
            item: torch.Tensor = stream.stream[name]
            if name in new_stream:
                new_stream[name].append(item)
            else:
                new_stream[name] = [item]

        # Transfer losses into buildern
        for name in stream.losses:
            loss: torch.Tensor = stream.losses[name]
            if name in new_losses:
                new_losses[name].append(loss)
            else:
                new_losses[name] = [loss]

        # transfer metric into builder
        for name in stream.metrics:
            metric: List[torch.Tensor] = stream.metrics[name]
            if name in new_metric:
                new_metric[name] += metric
            else:
                new_metric[name] = metric
    return new_stream, new_metric, new_losses

@torch.jit.script
def stream_concat(
                  streams: List[StreamTensor],
                  loss_reduce: str = "sum",
                  dim: int = -1
                  ) -> StreamTensor:
    """

    Combines the stream entries with an elementwise concatenation.

    :param streams: The streams to combine
    :param dim: The dimension to concat one
    :param loss_reduction: The means by which to combine duplicate
        loss entries. Options are sum, mean, max, min, and media
    :return: A StreamTensor
    """

    new_stream, new_losses, final_metrics = get_raw_merge(streams)

    final_stream = {}
    for name in new_stream:
        items: List[torch.Tensor] = new_stream[name]
        final: torch.Tensor = torch.concat(items, dim=dim)
        final_stream[name] = final

    final_losses = {}
    for name in new_losses:
        items: List[torch.Tensor] = new_losses[name]
        final: torch.Tensor = reduce(loss_reduce, items)
        final_losses[name] = final

    return StreamTensor(final_stream, final_losses, final_metrics, None)

def stream_merge(streams: List[StreamTensor],
                 stream_reduction: str = "sum",
                 loss_reduction: str = "sum",
                 )-> StreamTensor:
    """

    Merges together a collection of streams into a single
    primary stream. Combines their loss and metric information.

    :param streams: A list of streams to merge. The first is the primary branch,
        where the residuals will inherit from
    :param stream_reduction: The means by which to combine
        duplicate stream entries. Options are
        sum, mean, min, max, and median
    :param loss_reduction: The means by which to combine duplicate
        loss entries. Options are sum, mean, max, min, and media
    :return: A StreamTensor
    """
    new_stream, new_losses, final_metrics = get_raw_merge(streams)

    # Collapses the merges, where relevant
    final_stream = {}
    for name in new_stream:
        stream_entry: List[torch.Tensor] = new_stream[name]
        temp = reduce(stream_reduction, stream_entry)
        final_stream[name] = temp

    final_loss = {}
    for name in new_losses:
        loss_entry: List[torch.Tensor] = new_losses[name]
        temp = reduce(loss_reduction, loss_entry)
        final_loss[name] = temp

    # Return

    return StreamTensor(final_stream, final_loss, final_metrics, streams[0].residuals)

@torch.jit.script
def stream_split(stream: StreamTensor,
                 split_directions: Dict[str, List[int]],
                 dim: int = -1)-> List[StreamTensor]:
    """
    Isolates the given split definitions. Then performs splits as indicated
    :param stream: The tensor to split
    :param splits: The directions to split by. The string will be the split names,
        the list the dimensions
    :return: List[StreamTensors]
    """

    #Create updates
    update_names = list(split_directions.keys())
    update_stream: Dict[str, List[torch.Tensor]] = {}
    for name in update_names:
        assert name in stream.stream
        item =stream.stream[name]
        splits = item.split(split_directions[name], dim=dim)
        update_stream[name] = splits

    transposed_stream: List[Dict[str, torch.Tensor]] = [{name : torch.empty(0)} for name in update_names]
    for name in update_stream:
        layer = update_stream[name]
        for item, container in zip(layer, transposed_stream):
            container[name] = item

    updates = [StreamTensor(items) for items in transposed_stream]

    #Create streams
    outputs = [stream.discard(update_names) for _ in update_names]
    outputs = [stream_merge([stream, update]) for stream, update in zip(outputs, updates)]
    return outputs

@torch.jit.script
def zeros(definitions: Dict[str, List[int]]):
    """

    Constructs a series of tensors lying within a stream
    based on the indicated definitons

    :param definitions: names, and then tensors shapes
    :param dtype: the dtype
    :param device: the device
    :param requires_grad: whether we require grads
    :return:
    """

    stream = {}
    for name in definitions:
        value = torch.zeros(definitions[name])
        stream[name] = value
    return StreamTensor(stream, None, None, None)

