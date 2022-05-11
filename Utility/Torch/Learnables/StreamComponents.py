# perform imports
import math
from typing import Union, Sequence, Optional, Callable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F



# perform library imports
from Utility.Torch.Learnables import Layers
from Utility.Torch import Glimpses, Paddings



class


class LocalContext(nn.Module):
    """

    Performs local context for an incoming stream. This is similar to a variety of

    """

class PriorityStreamAttention(nn.Module):
    """

    Ask a sequence of questions. Get a sequence of answers.

    This accepts a list of streams and some sort of query. It then
    commits an efficient lookup to answer the questions

    Attention is performed with respect to each tier of the stream. critically,
    the score portion of the attention mechanism is then used to decide whether to
    continue working on the next portion of the stream.

    """
    @staticmethod
    def create_composite_stream(streams: List[List[torch.Tensor]]):
        streams = [list(x) for x in zip(*streams)]

        # Create the composite stream. This consists of an entity in which the various teirs
        # have been concatenated together, and the conversion parameters are retained. In particular
        # it consists of, for each entry, a tier that is the composite of all the inputs, and
        # for all but the last entry a expansion map, of same length as tier stream, which
        # tells how many values a particular index is summarizing.

        expansion_map = []
        refined_stream = []
        prior_lengths = None
        for tier in streams:

            lengths = []
            tier_content = []
            for j, stream in enumerate(tier):
                tier_content.append(stream)
                lengths.append(stream.shape[-2])

            # concat, store
            tier_content = torch.concat(tier_content, dim=-2)
            lengths = torch.Tensor(lengths)
            refined_stream.append(tier_content)
            # Calculate expansion ratio for each tier item
            if prior_lengths is not None:
                ratios = lengths / prior_lengths
                expansion = [torch.full([length], ratio) for length, ratio in zip(lengths, ratios)]
                expansion = torch.concat(expansion)
                expansion_map.append(expansion)
            prior_lengths = lengths
        return refined_stream, expansion_map
    def __init__(self,
                 d_query: int,
                 d_stream: int,
                 d_internal: int,
                 total_depth: int,
                 heads: int,
                 threshold: float,
                 attn_activation: str = "sigmoid",
                 unspool_source: str = "logits"):
        """

        :param d_query: how wide the query inputs will be
        :param d_stream: how wide each stream input will be
        :param d_internal: how wide we wish the internal logic channel to be
        :param total_depth: the maximum stream depth to setup projectors for
        :param heads: The number of projection heads to make
        :param threshold: The threshold for unspooling purposes
        :param attn_activation: The type of attention activation function. This applies
            after multiplying the key and query together. The three availble are
            "sigmoid", "softmax", and "raw", with raw using the raw logits and the
            rest being the standard functions
        :param unspool_source:
            The location that the unspooling mechanism gets its information from.
            The two options are "logits" and "score". "logits" threshold's the
            raw logits before they would run through the attn_activation. "score"
            runs the activation, then thresholds the result.

        """
        super().__init__()

        self._threshold = threshold
        self._total  = total_depth
        self._attn_mode = attn_activation
        self._unspooling_mode = unspool_source

        self._QueryProjectors = nn.ModuleList([Layers.Linear(d_query, [heads, d_internal]) for _ in range(total_depth)])
        self._KeyProjectors = nn.ModuleList([Layers.Linear(d_stream, [heads, d_internal]) for _ in range(total_depth)])
        self._ValueProjectors = nn.ModuleList([Layers.Linear(d_stream, [heads, d_query]) for _ in range(total_depth)])
        self._CollapseProjector = Layers.Linear([heads, d_query], d_query)


    def forward(self, streams: List[List[torch.Tensor]], query: torch.Tensor):
        """
        :param streams: A list of streams.
        :param query: A query. A torch tensor which it is wished to run against the stream.
        :return: result, loss
        """

        composite_stream, expansions = self.create_composite_stream(streams)
        index = torch.arange(composite_stream[0].shape[-2])
        output = torch.zeros_like(query)
        loss = torch.tensor(0.0)
        for i in range(len(composite_stream)):

            #Get content from the current tier. Only keep content which
            #matches the active index
            content = composite_stream[i]
            content = content[..., index, :]

            if content.shape[-2] == 0:
                #We have excluded all
                #valid indices. Stop early
                break

            #Get and perform the projections for the current layer. Create heads.
            #place output into (..., head, item, channel) format.
            Query_Projector = self._Query_Projectors[i]
            Key_Projector = self._KeyProjectors[i]
            Value_Projector = self._ValueProjectors[i]

            query = Query_Projector(query).transpose(-2, -3)
            key = Key_Projector(content).transpose(-2, -3)
            value = Value_Projector(content).transpose(-2, -3).unsqueeze(-1)

            #Perform attention. Three modes are defined. These are
            #sigmoid, softmax, and raw.
            logits = torch.matmul(query, key.transpose(-1, -2))
            if self._attn_mode == "sigmoid":
                score = torch.sigmoid(logits)
            elif self._attn_mode == "softmax":
                score = torch.softmax(logits, dim=-1)
            elif self._attn_mode == "raw":
                score = logits
            else:
                raise ValueError("Provided mode was not well defined")

            attn = torch.mul(score, value.transpose(-1, -2))
            attn = torch.matmul(score, value).squeeze(-1)


            if i + 1 < len(composite_stream):
                #If there remains tiers for next time, restrict which
                #ones are examined based on how active this time's tiers
                #were. Update the index accordingly.

                if self._unspooling_mode == "logits":
                    unspool = logits > self._threshold
                elif self._unspooling_mode == "score":
                    unspool = score > self._threshold
                else:
                    raise ValueError("Provided unspooling source is invalid")

                new_index = index.masked_select(unspool)
                expansion_ratios = expansions[i]
                expansion_ratios = expansion_ratios[index]

                new_index = [torch.arange(item, item+ratio) for item, ratio in zip(new_index, expansion_ratios)]
                new_index = torch.concat(new_index)
                index = new_index

            #Add results to output. Also, collect loss, which depends
            #entirely on how active the score logits are and particularly
            #on how many score logits are active.
            output = output+attn
            loss = loss + torch.relu(logits - self._threshold)
        return output, loss













        #Generate treemappings
        expansion_tier_map = [[]]*(len(streams) - 1)
        refined_content = [[]]*len(streams)
        for stream in enumerate(streams):

            #Create the extension t
            stream_map = []
            tier_content = []
            last_tier = None
            for j, tier_content in enumerate(stream):

                #Calculate the expansion ratio between the last and current tier
                if last_tier is None:
                    last_tier = tier_content
                    continue
                expansion_ratio = tier_content.shape[-2] //  last_tier.shape[-2]
                stream_map.append(torch.full([last_tier.shape[-2]], expansion_ratio, dtype=torch.float32))

                #Store away the current stream data
                refined_content[j].append(tier_content)

            stream_map = torch.concat(stream_map)
            expansion_tier_map.append(stream_map)



        #Transpose. Tier comes first

        transposed_stream = [list(x) for x in zip(*streams)]

        #Generate interconnectivity rates
        for











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
    def localized_component_attention(query: torch.Tensor,
                                      keys: List[torch.Tensor],
                                      values: List[torch.Tensor],
                                      kerneling: int = 0,
                                      blocking: int = 0,
                                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        Performs banded queried attention between a given query embedding and a list of
        key, value tensors which are related to the length of the query embedding by
        being found in multiples with respect to each other. In particular, the length
        of the key/value pairs must either be a multiple of the query, or a factor of the query.

        :param query: The query to attend with respect to
        :param keys: The list of keys to attend to
        :param values: The list of values to attend to
        :param kerneling: How much excess
        :return:
        """

        length = query.shape[-2]
        query = query.transpose(-1, -2)  # (..., data, item)
        processed_queries = []
        processed_keys = []
        processed_values = []
        for key, value in zip(keys, values):

            key = key.transpose(-1, -2)
            value = value.transpose(-1, -2)

            if length >= key.shape[-1]:
                assert length % key.shape[-1] == 0, "query length was not found to be a multiple of key length"
                repeat_rate = length // key.shape[-1]

                query_kernel = repeat_rate
                query_stride = repeat_rate

                key_kernel = 1 * (1 + blocking) + kerneling
                key_stride = 1

                value_kernel = 1 * (1 + blocking) + kerneling
                value_stride = 1

                total_padding = 1 * blocking + kerneling
                prepadding, postpadding = math.ceil(total_padding / 2), math.floor(total_padding / 2)
                pad_op = (prepadding, postpadding)

                key = F.pad(key, pad_op)
                value = F.pad(value, pad_op)
            else:
                assert key.shape[-1] % length == 0, "key length was not found to be a multiple of query length"
                repeat_rate = key.shape[-1] // length

                query_kernel = 1
                query_stride = 1

                key_kernel = repeat_rate * (blocking + 1) + kerneling
                key_stride = repeat_rate

                value_kernel = repeat_rate * (blocking + 1) + kerneling
                value_stride = repeat_rate

                total_padding = repeat_rate * blocking + kerneling
                prepadding, postpadding = math.ceil(total_padding / 2), math.floor(total_padding / 2)
                pad_op = (prepadding, postpadding)

                key = F.pad(key, pad_op)
                value = F.pad(value, pad_op)

            processed_queries.append(Glimpses.local(query, query_kernel, query_stride, 1))
            processed_keys.append(Glimpses.local(key, key_kernel, key_stride, 1))
            processed_values.append(Glimpses.local(value, value_kernel, value_stride, 1))

        ##Reorder placing data on last dimension
        permute = list(range(query.dim() + 1))
        moving = permute.pop(-3)
        permute.append(moving)

        processed_queries = [item.permute(permute) for item in processed_queries]
        processed_keys = [item.permute(permute) for item in processed_keys]
        processed_values = [item.permute(permute) for item in processed_values]

        output = []
        for query, key, value in zip(processed_queries, processed_keys, processed_values):

            # Query is items, local, data.
            # Key is items, local, data
            # Value is items, local, data

            score = torch.matmul(query, key.transpose(-1, -2))
            if mask is not None:
                score = score.masked_fill(mask, -1e9)
            score = torch.softmax(score, dim=-1)

            attn = torch.matmul(score, value)
            output.append(attn)

        # Are now in item, local, data again
        output = [item.flatten(-3, -2) for item in output]
        output = torch.concat(output, dim=-1)
        return output

    def __init__(self,
                 d_model: int,
                 stream_length: int,
                 heads: int = 3,
                 forward_interactivity: int = 1,
                 backward_interactivity: int = 1,
                 kerneling: int = 3,
                 blocking: int = 2,

                 ):
        super().__init__()

        self._kerneling = kerneling
        self._blocking = blocking

        #Calculate how to break apart the incoming data stream for best effect.
        #Save results for later use.
        self._startpoints = [max(0, i - backward_interactivity) for i in range(stream_length)]
        self._endpoints = [min(stream_length, i + 1 + forward_interactivity) for i in range(stream_length)]
        self._lengths = [high - low for high, low in zip(self._endpoints, self._startpoints)]

        #Create the necessary layers. These are a bunch of layers which will mostly be
        #applied in parallel

        self._QueryProjector = nn.ModuleList([Layers.Linear(d_model, [heads, d_model]) for _ in range(stream_length)])
        self._KeyProjector = nn.ModuleList([Layers.Linear(d_model, [heads, d_model]) for _ in range(stream_length)])
        self._ValueProjector = nn.ModuleList([Layers.Linear(d_model, [heads, d_model]) for _ in range(stream_length)])

        self._CollapseProjector = nn.ModuleList([Layers.Linear([heads, d_model*length], d_model) for length in self._lengths])

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
                attn = self.localized_component_attention(
                    query,
                    local_keys[i],
                    local_values[i],
                    self._kerneling,
                )
            else:
                attn = self.localized_component_attention(
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

