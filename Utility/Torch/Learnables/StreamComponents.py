# perform imports
import math
from typing import Union, Sequence, Optional, Callable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F



# perform library imports
from Utility.Torch import Glimpses, Paddings
from Utility.Torch.Learnables import EaseOfUse

class StreamConstructorCMHA(nn.Module):
    pass

class LocalContext(nn.Module):
    """

    Performs local context for an incoming stream. This is similar to a variety of

    """



class Interchange(nn.Module):
    """

    Interchange for a stream.

    This class performs banded interchange attention
    in a multiheaded fashion among the elements of an incoming component stream.

    Under this flavor of attention, the various stream items are expected to have
    lengths that are multiples of each other. For instance, in a ratio of 128:64:32:16.
    Other configurations will throw an error.

    Nearby elements in this stream perform banded attention to each other, in relation
    to the rate of item x to item y. For instance, with a 128:64 relationship, and the
    64 streamitem being the query entity, attention would be constructed with every 1 query item
    attending to every 2 key, value items. Additionally, a "localizing" parameter also allows going
    beyond this. For instance, in the same 128:64 example, a localizing setting of 2 would increase
    the kernel size by 2 such that attention would be constructed with query to key/value ratios of
    1:5. The extra content is provided by having the same items show up more than once.

    comparable properties:

        - Multiheaded Attention
        - Banding
        - Localization
        - Stream exchange
    """
    @staticmethod
    def component_attention(query: torch.Tensor,
                            keys: List[torch.Tensor],
                            values: List[torch.Tensor],
                            kerneling: int = 0,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        The attention step. Performs banded queried attention between
        the keys and values, then stacks and returns the results. Responsible
        for both examining local relationships and exchanging information between
        diverse tiers.

        :param query: The query to attend with respect to
        :param keys: The list of keys to attend to
        :param values: The list of values to attend to
        :param kerneling: How much extra to look at beyond the N:M mapping
            May detect local patterns.
        :return:
        """



        length = query.shape[-2]
        query = query.transpose(-1, -2) #(..., data, item)
        processed_queries = []
        processed_keys = []
        processed_values = []
        for key, value in zip(keys, values):

            key = key.transpose(-1, -2)
            value = value.transpose(-1, -2)

            if length > key.shape[-1]:
                #Query greater than key
                assert length % key.shape[-1] == 0, "query length was not found to be a multiple of key length"
                repeat_rate = length//key.shape[-1]

                key = F.pad(key, (math.ceil(kerneling/2), math.floor(kerneling/2)))
                value = F.pad(value, (math.ceil(kerneling/2), math.floor(kerneling/2)))

                #Setup scheme to reuse the query kernel.
                processed_queries.append(Glimpses.local(query, repeat_rate, repeat_rate, 1))
                processed_keys.append(Glimpses.local(key, kerneling+1, 1, 1))
                processed_values.append(Glimpses.local(value, kerneling+1, 1, 1))
            if length < key.shape[-1]:
                #Query less than key. Map many keys to one query
                assert key.shape[-1] % length == 0, "key length was not found to be a multiple of query length"
                repeat_rate = key.shape[-1]//length

                key = F.pad(key, (math.ceil(kerneling/2), math.floor(kerneling/2)))
                value = F.pad(value, (math.ceil(kerneling/2), math.floor(kerneling/2)))

                #We process one query with multiple keys, values
                processed_queries.append(Glimpses.local(query, 1, 1, 1))
                processed_keys.append(Glimpses.local(key, repeat_rate + kerneling, repeat_rate, 1))
                processed_values.append(Glimpses.local(value, repeat_rate + kerneling, repeat_rate, 1))

            if length == key.shape[-1]:
                #Query length equals key.
                key = F.pad(key, (math.ceil(kerneling/2), math.floor(kerneling/2)))
                value = F.pad(value, (math.ceil(kerneling/2), math.floor(kerneling/2)))


                processed_queries.append(Glimpses.local(query, 1, 1, 1))
                processed_keys.append(Glimpses.local(key, kerneling + 1, 1, 1))
                processed_values.append(Glimpses.local(value, kerneling +1, 1, 1))


        ##Reorder placing data on last dimension
        permute = list(range(query.dim() + 1))
        moving = permute.pop(-3)
        permute.append(moving)

        processed_queries = [item.permute(permute) for item in processed_queries]
        processed_keys = [item.permute(permute) for item in processed_keys]
        processed_values = [item.permute(permute) for item in processed_values]

        output = []
        for query, key, value in zip(processed_queries, processed_keys, processed_values):

            #Query is items, local, data.
            #Key is items, local, data
            #Value is items, local, data

            score = torch.matmul(query, key.transpose(-1, -2))
            if mask is not None:
                score = score.masked_fill(mask, -1e9)
            score = torch.softmax(score, dim=-1)

            attn = torch.matmul(score, value)
            output.append(attn)

        #Are now in item, local, data again
        output = [item.flatten(-3, -2) for item in output]
        output = torch.concat(output, dim=-1)
        return output

    def __init__(self,
                 d_model: int,
                 stream_length: int,
                 heads: int = 3,
                 forward_interactivity: int = 1,
                 backward_interactivity: int = 1,
                 localizing: int = 10,

                 ):
        super().__init__()

        self._kerneling = localizing

        #Calculate how to break apart the incoming data stream for best effect.
        #Save results for later use.
        self._startpoints = [max(0, i - backward_interactivity) for i in range(stream_length)]
        self._endpoints = [min(stream_length, i + 1 + forward_interactivity) for i in range(stream_length)]
        self._lengths = [high - low for high, low in zip(self._endpoints, self._startpoints)]

        #Create the necessary layers. These are a bunch of layers which will mostly be
        #applied in parallel

        self._QueryProjector = nn.ModuleList([Learnables.Linear(d_model, [heads, d_model]) for _ in range(stream_length)])
        self._KeyProjector = nn.ModuleList([Learnables.Linear(d_model, [heads, d_model]) for _ in range(stream_length)])
        self._ValueProjector = nn.ModuleList([Learnables.Linear(d_model, [heads, d_model]) for _ in range(stream_length)])

        self._CollapseProjector = nn.ModuleList([Learnables.Linear([heads, d_model*length], d_model) for length in self._lengths])

    def forward(self, component_stream: List[torch.Tensor],
                mask: Optional[List[torch.Tensor]] = None):
        """


        :param component_stream:
        :return:
        """

        #Perform query, key, value projections and add heads.
        queries = []
        keys = []
        values = []

        permute = list(range(component_stream[0].dim() + 1))
        moving = permute.pop(-2)
        permute.insert(-2, moving)
        for i, stream_item in enumerate(component_stream):
            queries.append(self._QueryProjector[i](stream_item).permute(permute))
            keys.append(self._KeyProjector[i](stream_item).permute(permute))
            values.append(self._ValueProjector[i](stream_item).permute(permute))



        #Create local subrepresentations
        local_keys = []
        local_values = []
        for startpoint, endpoint in zip(self._startpoints, self._endpoints):
            local_keys.append(keys[startpoint:endpoint])
            local_values.append(values[startpoint:endpoint])

        #Perform attention. Get results
        output = []
        for i, query in enumerate(queries):
            collapse = self._CollapseProjector[i]

            if mask is None:
                attn = self.component_attention(
                    query,
                    local_keys[i],
                    local_values[i],
                    self._kerneling,
                )
            else:
                attn = self.component_attention(
                    query,
                    local_keys[i],
                    local_values[i],
                    self._kerneling,
                    mask[i]
                )
            #Reduce head and crosstalk away

            attn = attn.transpose(-2, -3)
            attn = collapse(attn)
            output.append(attn)
        return output