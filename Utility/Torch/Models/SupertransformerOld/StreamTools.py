
from __future__ import annotations

from typing import List, Optional, Union, Tuple, Dict

import torch

"""

As the complexity of the models I am working with has grown,
using tuples to track information is becoming increasingly
untenable. Any minor modification requires major changes to all layers.

As a result, I now utilize the stream mechanism. This involves placing
information into a dictionary, known as the stream, from which it can be
later retrieved. 

A stream consists of a collection of tensors, losses, and metrics which
travel together throughout a model. Generally, there exists a primary stream,
and a sequence of alternative streams.
"""


###### Traditional ######
class _LeafCollection():
    """
    Entirely function. This is a collection
    of tensors, and a leaf of a stream tree.
    """
    ##core functional update methods.

    def clone(self, parent: _StreamTensor)-> _LeafCollection:
        items: List[torch.Tensor] = []
        for item in self._collection:
            items.append(item.clone())
        return _LeafCollection(parent, self.name, items)
    def set_all(self, tensors: List[torch.Tensor]) -> _StreamTensor:
        return self.parent.set({self.name : tensors})
    def commit_reduce(self, tensor: torch.Tensor) -> _StreamTensor:
        return self.parent.set({self.name : tensor})

    #Editing methods. Completely functional. Attemts to set new
    #instance to higher tree.

    def set(self, index: int,  tensor: torch.Tensor) -> _StreamTensor:
        new_tensors = self._collection.copy()
        new_tensors[index] = tensor
        return self.set_all(new_tensors)
    def append(self, tensor: torch.Tensor)-> _StreamTensor:
        tensors = self._collection.copy()
        tensors.append(tensor)
        return self.set_all(tensors)
    def insert(self, index: int, value: torch.Tensor) -> _StreamTensor:
        tensors = self._collection.copy()
        tensors.insert(index, value)
        return self.set_all(tensors)
    def pop(self, index: Optional[int] = None) -> Tuple[torch.Tensor, _StreamTensor]:
        tensors = self._collection.copy()
        if index is None:
            item = tensors.pop()
        else:
            item = tensors.pop(index)
        return item, self.set_all(tensors)
    #Retrieval.
    def get(self, index: int, default: Optional[torch.Tensor] =None)-> torch.Tensor:
        if len(self._collection) < index:
            return default
        return self._collection[index].clone()
    def get_all(self)-> List[torch.Tensor]:
        output = []
        for item in self._collection:
            output.append(item.clone())
        return output
    def __add__(self, other: Union[_LeafCollection, _LeafTensor])-> _StreamTensor:
        tensors = self._collection.copy()
        if isinstance(other, _LeafTensor):
            tensors.append(other.tensor)
        elif isinstance(other, _LeafCollection):
            tensors = tensors + other.get_all()
        else:
            raise RuntimeError("Unknown type provided")
        return self.set_all(tensors)
    def __radd__(self, other: Union[_LeafCollection, _LeafTensor]) -> _StreamTensor:
        tensors = self._collection.copy()
        if isinstance(other, _LeafTensor):
            tensors.insert(0, other.tensor)
        elif isinstance(other, _LeafCollection):
            tensors = other.get_all() + tensors
        else:
            raise RuntimeError("Unknown type provided")
        return self.set_all(tensors)



    #Magic
    def __getitem__(self, item: int) -> torch.Tensor:
        return self._collection[item].clone()
    def __contains__(self, item) -> bool:
        return item in self._collection
    def __len__(self) -> int:
        return len(self._collection)
    def __iter__(self) -> List[torch.Tensor]:
        return self._collection
    def __eq__(self, other: _LeafCollection) -> bool:
        if len(self) != len(other):
            return False
        for my_item, their_item in zip(self.__iter__(), other.__iter__()):
            if not torch.all(my_item == their_item):
                return False
        return True

    ### Reduction and conversion ###

    def _reduce_broadcast(self, items: List[torch.Tensor])-> List[torch.Tensor]:
        """ Attempts to broadcast items together such the shapes are common"""

        # Figure out what the broadcast shape should be
        target_shape: List[int] = []
        source_location: int = 0
        for i, item in enumerate(items):
            if len(item.shape) > len(target_shape):
                target_shape = list(item.shape)
                source_location = i

        #Validate broadcast. Explain issue if present
        for i, item in enumerate(items):
            required_shape = target_shape[-len(item.shape):]
            current_shape = list(item.shape)
            if required_shape != current_shape:
                msg = "merge collection at location %s was not compatible with collection at %s" % (i, source_location)
                raise ValueError(msg, required_shape, current_shape)

        # Perform broadcast. Return
        output: List[torch.Tensor] = []
        for i, item in enumerate(items):
            new_item = torch.broadcast_to(item, target_shape)
            output.append(new_item)
        return output
    def _concat_broadcast(self,
                         items: List[torch.Tensor],
                         concat_dim: int = -1):
        """ Broadcasts for concat. Ensures all dimensions but concat dim are same shape"""
        assert concat_dim < 0, "For numerical reasons, concat_dim must be < 0"
        broadcast_shape: List[int] = []
        for item in items:
            if len(item.shape) - 1 > len(broadcast_shape):
                new_shape = list(item.shape)
                new_shape.pop(concat_dim)
                broadcast_shape = new_shape

        output: List[torch.Tensor] = []
        breakapart_dim = concat_dim % len(broadcast_shape) + 1

        for item in items:
            new_shape = broadcast_shape.copy()
            prior_shape, post_shape = new_shape[:breakapart_dim], new_shape[breakapart_dim:]
            prior_shape.append(item.shape[concat_dim])
            new_shape = prior_shape + post_shape
            new_item = torch.broadcast_to(item, new_shape)
            output.append(new_item)
        return output

    def sum_reduce(self) -> _StreamTensor:
        """ Attempts to sum up everything in the collection. Converts it to a tensor"""
        broadcast = self._reduce_broadcast(self._collection)
        items = torch.stack(broadcast, dim=0)
        items = items.sum(dim=0)
        return self.commit_reduce(items)
    def mean_reduce(self) -> _StreamTensor:
        """ Attempts to average up everything in the collection"""
        broadcast = self._reduce_broadcast(self._collection)
        items = torch.stack(broadcast, dim=0)
        items = items.mean(dim=0)
        return self.commit_reduce(items)
    def max_reduce(self) -> _StreamTensor:
        """ Attempts to find the max of everything in the collection"""

        broadcast = self._reduce_broadcast(self._collection)
        items = torch.stack(broadcast, dim=0)
        items, _ = items.min(dim=0)
        return self.commit_reduce(items)
    def min_reduce(self) -> _StreamTensor:
        """ Attempts to find the min of everything in the collection"""
        broadcast = self._reduce_broadcast(self._collection)
        items = torch.stack(broadcast, dim=0)
        items, _ = items.max(dim=0)
        return self.commit_reduce(items)
    def median_reduce(self) -> _StreamTensor:
        """ Attempts to find the median of everything in the collection"""
        broadcast = self._reduce_broadcast(self._collection)
        items = torch.stack(broadcast, dim=0)
        items, _ = items.median(dim=0)
        return self.commit_reduce(items)
    def concat_reduce(self):
        """ Attempts to concatenate together everything in the collection"""
        broadcast = self._concat_broadcast(self._collection)
        items = torch.concat(broadcast, dim=-1)
        return self.commit_reduce(items)
    def __init__(self,
                 parent: _StreamTensor,
                 name: str,
                 collection: List[torch.Tensor]):
        self.name = name
        self._collection = collection
        self.parent = parent


class _LeafTensor():
    """
    The tensor class contains a tensor,
    and allows manipulation of tensors
    """


    @property
    def value(self)-> torch.Tensor:
        return self.tensor
    def set(self, tensor: torch.Tensor)-> _StreamTensor:
        return self._parent.set({self.name: tensor})
    def collection(self)-> _StreamTensor:
        return self._parent.set({self.name : [self.value]})
    def clone(self, parent: _StreamTensor)-> _LeafTensor:
        return _LeafTensor(parent, self.name, self.tensor)
    def __init__(self,
                 parent: _StreamTensor,
                 name: str,
                 tensor: torch.Tensor):
        self.name = name
        self.tensor = tensor
        self._parent = parent

class _StreamTensor():
    """
    A stream tensor designed to hold a lot of items elegantly
    at once, and enable easy manipulation of such.

    Acts as a node in a tree. The tree nodes may terminate in one
    of two ways: as a single tensor, or a list of tensor. Attempting
    to set the class with a particular name will create a node
    of the appropriate type.
    """
    def __getattr__(self, item):
        if item in self._names:
            item = super().__getattribute__(item)


    #Core functional items. Some things here are inplace. Nothing else is.
    def clone(self,
              exclude: Optional[List[str]] = None,
              include: Optional[List[str]] = None,
              parent: Optional[_StreamTensor] = None) -> _StreamTensor:
        """
        Create an independent clone of this and every node below it.
        """
        items = {name: getattr(self, name) for name in self._names}
        final_items: Dict[str, Union[_StreamTensor, torch.Tensor, List[torch.Tensor]]] = {}
        for name in items:
            if exclude is not None and name in exclude:
                continue
            if include is not None and name not in include:
                continue
            item = items[name]
            if isinstance(item, _StreamTensor):
                final_items[name] = item
            elif isinstance(item, _LeafCollection):
                final_items[name] = item.get_all()
            elif isinstance(item, _LeafTensor):
                final_items[name] = item.tensor
            else:
                raise RuntimeError("Illegal state reached")
        return _StreamTensor(self.name, items, parent)
    def set(self, items: Dict[str, Union[torch.Tensor, List[torch.Tensor], _StreamTensor]]) -> \
            _StreamTensor:
        """
        Sets the indicated entries in items to the appropriate values.
        Then returns the entire rebuilt tree.
        :param items: the items to set. May be tensors, a collection of tensors, or
            even another streamtensor.
        :return:
        """
        new_items = {name: getattr(self, name) for name in self._names}
        for name, value in items.items():
            new_items[name] = value
        new_node = _StreamTensor(self.name, new_items)
        if self.parent is None:
            return new_node
        else:
            return self.parent.set({self.name : new_node})

    def __init__(self,
                 name: str,
                 items: Optional[Dict[str, Union[torch.Tensor, List[torch.Tensor], _StreamTensor]]] = None):

        if items is None:
            final_items = {}
        else:
            final_items = items

        self.name = name
        self._names = []

        for name, value in final_items.items():
            if isinstance(value, torch.Tensor):
                item = _LeafTensor(self, name, value)
                setattr(self, name, item)
            elif isinstance(value, list):
                item = _LeafCollection(self, name, value)
                setattr(self, name, item)
            elif isinstance(value, _StreamTensor):
                setattr(self, name, value.clone(parent = self))


@torch.jit.script
class StreamTensor():
    """

    A stream consists of a key-value map with an arbitrary
    number of named tensors, and an entry called "losses"
    which is another dictionary.

    The current branch is a collection of losses and metrics
    which have been accumulated under the current stream. The
    method "branch" can make a new branch, and will reset these.
    """

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
    def clone(self, include_stream: bool =True,
              include_losses: bool=True,
              include_metrics: bool=True,
              include_collections: bool = True)-> StreamTensor:
        """
        Creates a structually independent clone of the stream
        :return: A clone of the dictionary, with the dictionary itself being independent, but
        the entries being the same. Modifying the data structure does not modify the original
        """

        stream = None
        losses = None
        metrics = None
        collections = None



        if include_losses and include_metrics:
            return StreamTensor(self._stream, self._losses, self._metrics, self._residuals)
        if include_losses:
            return StreamTensor(self._stream, self._losses, None, None)
        if include_metrics:
            return StreamTensor(self._stream, None, None, None)
        return StreamTensor(self._stream, None, None, None)
    #Element manipulation
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
        Discards from the stream the items with the indicated names. This
        keeps the current branch

        :param names: The names to discard
        :return: A StreamTensor
        """
        new_stream = self.stream.copy()
        for name in names:
            assert name in new_stream
            new_stream.pop(name)

        return StreamTensor(new_stream, self.losses, self.metrics, self._residuals)
    def keeponly(self, names: List[str])-> StreamTensor:
        """

        Keeps only the items with the indicated names. This maintains the current
        branch

        :param names:
        :return: StreamTEnsor
        """

        new_stream = {}
        for name in names:
            assert name in self.stream
            new_stream[name] = self.stream[name]

        return StreamTensor(new_stream, self.losses, self.metrics, self._residuals)
    def split(self,
              split_directive: Dict[str, Tuple[int, List[int]]],
              include_unnamed: bool = True,

              ) -> List[StreamTensor]:
        """
        Directs a parallel split to occur using the indicated split directives.
        Only the first of these will retain its losses and metrics.


        split_directive is an instruction on how to perform the split. Should consists of a dictionary
        with the keys being the names, and the entry being a tuple. Each Tuple should
        first consist of an int, representing the dimension to split on, and then
        a list of ints, representing how long to make each split section.

        The size of each List[int] object must be the same, and determines how
        many StreamTensors are returned

        include_unnamed determines whether or not the none-named stream entries
        are discarded.

        :param split_directives:
        :return: A list of StreamTensors
        """

        split_items: Dict[str, List[torch.Tensor]] = {}
        list_length = 0
        for name in split_directive:
            assert name in self.stream
            item = self.stream[name]
            dim, directions = split_directive[name]
            split_item: List[torch.Tensor] = item.split(directions, dim=dim)
            if list_length == 0:
                list_length = len(split_item)
            elif list_length != len(split_item):
                raise ValueError("Attempt to split using unequal lists")
            split_items[name] = split_item

        output_collection: List[StreamTensor] = []
        for i in range(list_length):
            stream_data: Dict[str, torch.Tensor] = {}
            for name in self.stream:
                if name in split_items:
                    stream_data[name] = split_items[name][i]
                elif include_unnamed:
                    stream_data[name] = self.stream[name]
            if i == 0:
                streamtensor = StreamTensor(stream_data, self.losses, self.metrics, None)
            else:
                streamtensor = StreamTensor(stream_data, None, None, None)
            output_collection.append(streamtensor)

        return output_collection

    def __getitem__(self, item: str) -> torch.Tensor:
        return self.stream[item]
    def __repr__(self):
        stream_dict = {name: str(value.shape)  for name, value in self.stream.items()}

        rep = '< StreamTensor |'
        rep = rep + 'Streams : %s' % stream_dict
        rep = rep + '| Losses %s' % self.losses.keys()
        rep = rep + '| Metrics %s' % self.metrics.keys()
        rep = rep + '>'
        return str(rep)
    def __init__(self,
                 stream: Optional[Dict[str, torch.Tensor]] = None,
                 losses: Optional[Dict[str, torch.Tensor]] = None,
                 metrics: Optional[Dict[str, List[torch.Tensor]]] = None,
                 collections: Optional[Dict[str, List[torch.Tensor]]] = None, ):

        if stream is None:
            stream = {}
        if losses is None:
            losses = {}
        if metrics is None:
            update: Dict[str, List[torch.Tensor]] = {}
        else:
            update = metrics
            torch.jit.annotate(Dict[str, List[torch.Tensor]], update)

        if collections is None:
            update2: Dict[str, List[torch.Tensor]] = {}
        else:
            update2 = collections
            torch.jit.annotate(Dict[str, List[torch.Tensor]], update2)

        self._stream = stream
        self._losses = losses
        self._metrics = update
        self._residuals = update2


## Stream Merging and Helpers ###

class _MergeHelper():
    """
    A helper class. A holder for the functions which can be
    applied to a list to prep for the final tensor
    """
    def __init__(self, items: Dict[str, List[torch.Tensor]]):
        self.reduced = False
        self.initial: Dict[str, List[torch.Tensor]] = items
        self.final: Dict[str, torch.Tensor] =  {}
    def reduce_broadcast(self, items: List[torch.Tensor])-> List[torch.Tensor]:
        """ Attempts to broadcast items together such the shapes are common"""

        # Figure out what the broadcast shape should be
        target_shape: List[int] = []
        source_location: int = 0
        for i, item in enumerate(items):
            if len(item.shape) > len(target_shape):
                target_shape = list(item.shape)
                source_location = i

        #Validate broadcast. Explain issue if present
        for i, item in enumerate(items):
            required_shape = target_shape[-len(item.shape):]
            current_shape = list(item.shape)
            if required_shape != current_shape:
                msg = "merge collection at location %s was not compatible with collection at %s" % (i, source_location)
                raise ValueError(msg, required_shape, current_shape)

        # Perform broadcast. Return
        output: List[torch.Tensor] = []
        for i, item in enumerate(items):
            new_item = torch.broadcast_to(item, target_shape)
            output.append(new_item)
        return output
    def concat_broadcast(self,
                         items: List[torch.Tensor],
                         concat_dim: int = -1):
        """ Broadcasts for concat. Ensures all dimensions but concat dim are same shape"""
        assert concat_dim < 0, "For numerical reasons, concat_dim must be < 0"
        broadcast_shape: List[int] = []
        for item in items:
            if len(item.shape) - 1 > len(broadcast_shape):
                new_shape = list(item.shape)
                new_shape.pop(concat_dim)
                broadcast_shape = new_shape

        output: List[torch.Tensor] = []
        breakapart_dim = concat_dim % len(broadcast_shape) + 1

        for item in items:
            new_shape = broadcast_shape.copy()
            prior_shape, post_shape = new_shape[:breakapart_dim], new_shape[breakapart_dim:]
            prior_shape.append(item.shape[concat_dim])
            new_shape = prior_shape + post_shape
            new_item = torch.broadcast_to(item, new_shape)
            output.append(new_item)
        return output
    def sum(self) -> None:
        """ Reduce by sum this attribute"""
        assert self.reduced is False, "Already reduced"
        final = {}
        for name in self.initial:
            item = self.reduce_broadcast(self.initial[name])
            item = torch.stack(item, dim=0)
            item = item.sum(dim=0)
            final[name] = item
        self.final = final
        self.reduced = True
    def mean(self) -> None:
        """ Reduce by mean this attribute"""
        assert self.reduced is False, "Already reduced"
        final = {}
        for name in self.initial:
            item = self.reduce_broadcast(self.initial[name])
            item = torch.stack(item, dim=0)
            item = item.mean(dim=0)
            final[name] = item
        self.final = final
        self.reduced = True
    def max(self)-> None:
        """Reduce by max this attribute"""
        assert self.reduced is False, "Already reduced"
        final = {}
        for name in self.initial:
            item = self.reduce_broadcast(self.initial[name])
            item = torch.stack(item, dim=0)
            item, _ = item.max(dim=0)
            final[name] = item
        self.final = final
        self.reduced = True
    def min(self)-> None:
        """Reduce by min this attribute"""
        assert self.reduced is False, "Already reduced"
        final = {}
        for name in self.initial:
            item = self.reduce_broadcast(self.initial[name])
            item = torch.stack(item, dim=0)
            item, _ = item.min(dim=0)
            final[name] = item
        self.final = final
        self.reduced = True
    def median(self)-> None:
        """Reduce by median this attribute"""
        assert self.reduced is False, "Already reduced"
        final = {}
        for name in self.initial:
            item = self.reduce_broadcast(self.initial[name])
            item = torch.stack(item, dim=0)
            item, _ = item.median(dim=0)
            final[name] = item
        self.final = final
        self.reduced = True
    def first(self)-> None:
        """Reduce by keeping only the first attribute"""
        assert self.reduced is False, "Already reduced"
        final = {}
        for name in self.initial:
            final[name] = self.initial[name][0]
        self.final = final
        self.reduced = True
    def reduce_mode(self, directive: str):
        """ Allows a string to indicate reduction type

        Allowed options are "sum", "mean", "min", "max", "median", and "first"
        """
        assert directive in ("sum", "mean", "min", "max", "median", "first")
        if directive == "sum":
            self.sum()
        elif directive == "mean":
            self.mean()
        elif directive == "min":
            self.min()
        elif directive == "max":
            self.max()
        elif directive == "median":
            self.median()
        elif directive == "first":
            self.first()

    def concat(self, dim: int=-1)-> None:
        """ Reduce by concat this attribute"""
        assert self.reduced is False, "Already reduced"
        final = {}
        for name in self.initial:
            item = self.concat_broadcast(self.initial[name], dim)
            item = torch.concat(item, dim=dim)
            final[name] = item
        self.final = final
        self.reduced = True
    def __repr__(self)-> Union[
        Dict[str, torch.Tensor],
        Dict[str, List[torch.Tensor]]
    ]:
        if self.reduced:
            return self.final
        else:
            return self.initial


@torch.jit.script
class StreamMerger():
    """
    A collection specifically built for merging stream tensors.

    Upon merge, ambiguity is introduced when multiple streams carry
    the same key. This must be dealt with in the merge tensor. The
    merge tensor contains in it's stream and losses slots a class
    which can be told how to resolved the ambiguity, per slot. For instance, the followng

    mergetensor = MergeTensor(stream_collection)
    mergetensor.stream.sum()
    mergetensor.losses.mean()

    would resolve this by telling the stream items to sum, and the loss items to mean.
    Furthermore, one may then build back to a stream tensor with build.

    merged_stream_tensor = mergetensor.build()


    Options for reduction are sum, mean, min, max, median, first, and concat.



    """

    @property
    def raw_stream(self):
        return self._stream.initial
    @property
    def raw_losses(self):
        return self._losses.initial

    @property
    def stream(self)-> _MergeHelper:
        return self._stream
    @property
    def losses(self)-> _MergeHelper:
        return self._losses
    @property
    def metrics(self):
        return self._metrics

    ## Helper Functions ##
    def raw_merge(self, streams: List[StreamTensor]) -> \
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
    def build(self):
        """ Builds a new stream tensor once the ambiguity is resolved"""
        assert self._stream.reduced, "Stream was never reduced"
        assert self._losses.reduced, "Losses was never reduced"
        return StreamTensor(self._stream.final, self._losses.final, self.metrics)

    def __init__(self, streams: List[StreamTensor]):

        stream, metrics, losses = self.raw_merge(streams)

        self._stream: _MergeHelper = _MergeHelper(stream)
        self._losses: _MergeHelper = _MergeHelper(losses)
        self._metrics = metrics



### Stream editor plus helper classes ###

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

