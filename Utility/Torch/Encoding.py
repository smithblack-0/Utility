"""





"""

import torch
import numbers
import math
### Positional Encoding Utility. Caching, abstract, slicing. ###

class TrigPosEncoding():
    """
    Description:

    An automatic resizing, caching, slicable positional encoding class. It uses trig encodings

    After initialization, one may access the encodings
    using  Encoding[period, channel_max, indexSlice, channelSlice].

    This returns a slice over the targetted indices. Notably, negative indices
    are explicitly supported: One can ask for negative encodings, such as entries
    prior to a item. Also of note, if possible the class will reuse the current
    buffer for a given period, channel_max pair.

    channel_max, a new item, is not a hard max per say, but rather a constant that
    one will not greatly exceed in terms of channel length. Doing so will tend to
    lead to out of range errors.


    """

    class offsetTensor():

        @property
        def shape(self):

            low = torch.zeros(len(self._tensor.shape), dtype=torch.int32) - self._offset
            high = torch.tensor(self._tensor.shape, dtype=torch.int32) - self._offset
            return torch.stack([low, high], dim=-1)

        @property
        def offset(self):
            return self._offset

        @offset.setter
        def offset(self, value):
            assert len(self._tensor.shape) == len(value)
            self._offset = value

        def __init__(self, tensor, offsets):
            self._tensor = tensor
            self._offset = torch.tensor(offsets, dtype=torch.int32)
            assert len(tensor.shape) == len(offsets), "Offsets and tensor length must match"

        def __getitem__(self, key):

            if isinstance(key, numbers.Number):
                key = [slice(key, key + 1, 1)]

            final_key = []
            for item in key:
                if isinstance(item, numbers.Number):
                    final_key.append(slice(item, item + 1, 1))
                else:
                    final_key.append(item)

            key = final_key

            # Apply offsets
            final_slice = []
            for item, offset in zip(key, self._offset):
                if item.start is None:
                    start = item.start
                else:
                    start = item.start + offset

                if item.stop is None:
                    stop = item.stop
                else:
                    stop = item.stop + offset

                final_slice.append(slice(start, stop, item.step))

            # Get result
            return self._tensor[final_slice]

    @classmethod
    def make(cls, period, channel_modifier, minimums, maximums):
        """

        This function makes a buffer.

        The function makes a buffer with entries between
        minimums and maximums for each of the two axis in minimum
        and maximum.

        It then returns a tensor in offset format, which
        can be used for lookups

        """

        minimums = torch.tensor(minimums)
        maximums = torch.tensor(maximums)

        # Create indices,  ready for broadcast, and define output space
        position_indices = torch.arange(minimums[0], maximums[0]).unsqueeze(-1)
        channel_indices = torch.arange(minimums[1], maximums[1]).unsqueeze(0)

        requirements = tuple([*(maximums - minimums).numpy()])
        output = torch.zeros(requirements)

        # Create primary encoding, then embed sine, cosine outputs into the result.
        primary_encodings = 2 * math.pi * position_indices * (1 / period) ** (channel_indices / channel_modifier)

        # Account for the fact that zero should remain sine-encoded no matter what
        if primary_encodings.shape[0] % 2 == 0:
            output[::2] = torch.sin(primary_encodings[::2])
            output[1::2] = torch.cos(primary_encodings[1::2])
        else:
            output[1::2] = torch.sin(primary_encodings[1::2])
            output[::2] = torch.cos(primary_encodings[::2])
        # Build, and return, the result

        return cls.offsetTensor(output, -minimums)

    def __getitem__(self, key):
        """

        The primary access point for this instance.

        Expects to be fed a index or slice object of format
        [period, position, channel]. Position and channel may be
        slices, in which case they MUST contain a well defined
        start and end point.

        period must be a constant. There will be an additional buffer
        for every provided, hashable period.

        """

        # Unpacking and basic sanity checking.

        period, channel_modifier, position, channel = key

        assert isinstance(period, numbers.Number)
        assert isinstance(channel_modifier, numbers.Number)

        if isinstance(position, numbers.Number):
            position = slice(position, position + 1, 1)
        if isinstance(channel, numbers.Number):
            channel = slice(channel, channel + 1, 1)

        hash_item = (period, channel_modifier)

        # Gather important boundary information

        min_boundary = []
        max_boundary = []

        for item in [position, channel]:
            min_boundary.append(item.start)
            max_boundary.append(item.stop)

        needed_min_dimension = torch.tensor(min_boundary)
        needed_max_dimension = torch.tensor(max_boundary)

        # Rebuild the buffer if needed
        if hash_item not in self.cache:
            self.cache[hash_item] = self.make(period, channel_modifier, needed_min_dimension, needed_max_dimension)

        tensor = self.cache[hash_item]
        lowest_shape, highest_shape = tensor.shape[:, 0], tensor.shape[:, 1]
        if torch.any(lowest_shape > needed_min_dimension) or torch.any(highest_shape < needed_max_dimension):
            # Caculate the new minimum and maximum shape boundaries. Do this by keeping the lowest minimums,
            # and highest maximums, between the minimum and maximum pairs.

            cross_required_minimums = torch.tensor(np.minimum(lowest_shape, needed_min_dimension))
            cross_required_maximums = torch.tensor(np.maximum(highest_shape, needed_max_dimension))
            # New minimums, to encapsulate slice

            # Rebuild the buffer
            self.cache[hash_item] = self.make(period, channel_modifier, cross_required_minimums,
                                              cross_required_maximums)

        # Access then lookup.
        tensor = self.cache[hash_item]
        return tensor[position, channel]

    def __init__(self):
        """
        Start Up
        """

        self.cache = {}