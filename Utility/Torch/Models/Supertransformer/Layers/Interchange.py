"""

Layers responsible for exchanging information between various datastreams,
such as memory and text, memory and bus, etc, live here.


"""

from typing import List, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers


class Interchange(nn.Module):
    """

    A memory exchange layer. Accepts a
    text stream and memory argument. Returns
    a new text_stream and memory argument.

    Itself responsible Principly for memory exchange
    and outer normalization. Provided to it will be
    layer stacks responsible for attention and feedforward steps.

    Internal flow according to:
    https://docs.google.com/drawings/d/1Ej0ZlPbTqyDC_aC1xiMwghGn28IDDC8667dgyHr-I0Y/edit
    """

    def __init__(self,
                 d_stream: int,
                 d_memory: int,
                 stream_processing: nn.Module,
                 memory_processing: nn.Module,
                 transfer_heads: int = 2,
                 dropout: float = 0.1,
                 ):
        """

        :param d_stream: Width of stream embeddings
        :param d_memory: Width of memory embeddings
        :param stream_processing: A module which handles anything like self attn, conv, and feedforward.
        :param memory_processing: A module which handles self attn and feedforward
        :param transfer_heads: How many heads the interchange has. Default of 2
        :param dropout: Dropout rate. default of 0.1
        """
        super().__init__()
        #Store assertion info

        self._d_memory = d_memory
        self._d_stream = d_stream

        # Store processing
        self._stream_processing = torch.jit.script(stream_processing)
        self._memory_processing = torch.jit.script(memory_processing)

        # Create norms
        self._stream_intake_norm = nn.LayerNorm(d_stream)
        self._memory_intake_norm = nn.LayerNorm(d_memory)

        self._stream_output_norm = nn.LayerNorm(d_stream)
        self._memory_output_norm = nn.LayerNorm(d_memory)

        # Create exchange attentions.
        self._stream_to_memory_attn = nn.MultiheadAttention(d_memory,
                                                            transfer_heads,
                                                            dropout,
                                                            kdim=d_stream,
                                                            vdim=d_stream)
        self._memory_to_stream_attn = nn.MultiheadAttention(d_stream,
                                                            transfer_heads,
                                                            dropout,
                                                            kdim=d_memory,
                                                            vdim=d_memory)

    def forward(self, text_stream: torch.Tensor, memory: torch.Tensor):

        assert text_stream.shape[-1] == self._d_stream
        assert memory.shape[-1] == self._d_memory

        # Normalize input stream, then transfer stream info into memory.
        text_stream = self._stream_intake_norm(text_stream)
        attn = self._stream_to_memory_attn(memory, text_stream, text_stream)
        memory = self._memory_intake_norm(attn + memory)

        # Perform processing, including layernorm. Include residual bypass

        text_stream = self._stream_output_norm(self._stream_processing(text_stream) + text_stream)
        memory = self._memory_output_norm(self._memory_processing(memory))

        # Transfer memory information into stream, then return values
        text_stream = self._memory_to_stream_attn(text_stream, memory, memory)
        return text_stream, memory