"""

An encoder sequence is used to produce a sequence of ratios
with respect to the original input length consisting of
a_0, a_1, a_2, etc, where a_0 is equal to different lengths,
and a_f is global


"""
from typing import List

import torch
from torch import nn
from Utility.Torch import Learnables
from Utility.Torch.Archive import Architecture


class AttentionSublayer(nn.Module):
    """

    """


class StreamProcessingLayer(nn.Module):
    """
    Contains the memory exchange logic and residual bypass functions.

    Inside this layer is defined the memory exchange logic. The first thing that will happen
    is that prior residuals will be added together, and then normalized. Following this, an
    exchange is made transfering stream information into the memory. The provided
    sequence of text_stream_processing and memory_stream_processing layers are then called.

    Finally, the memory is used to condition the text stream, the residuals are generated, and returned

    """
    class AddNorm(nn.Module):
        def __init__(self, d_model):
            super().__init__()

            self._norm = nn.LayerNorm(d_model)
        def forward(self, *args):
            output = torch.stack(args, dim=0).sum(dim=0)
            output = self._norm(output)
            return output

    class SubLayer(nn.Module):
        def __init__(self, module, d_model):

            super().__init__()

            self._module = module
            self._norm = nn.LayerNorm(d_model)
        def forward(self, tensor: torch.Tensor, *args):
            """
            Only passes forward the first input

            :param tensor:
            :param args:
            :return:
            """
            output = self._norm(tensor)
            output = self._module(tensor, *args)
            output = output + tensor
            return output

    def __init__(self,
                 d_stream: int,
                 d_memory: int,
                 heads: int,
                 text_stream_processing: List[nn.Module],
                 memory_processing: List[nn.Module],
                 ):
        """

        :param text_stream_processing: A sequence of layers, called to process the traveling text stream
        :param memory_processing: A sequence of layers, called to process the traveling memory sequence.
        """
        super().__init__()

        #Create the intake layernorms

        self._intake_stream_norm = self.AddNorm(d_stream)
        self._intake_memory_norm = self.AddNorm(d_memory)
        self._transfer_into_memory_norm = nn.LayerNorm(d_memory)

        #Create the stack layernorms, and the process stacks. Do NOT wrap the last layer in the sequence


        self._stream2memory = nn.MultiheadAttention(d_memory, heads, 0.1, kdim=d_stream, vdim=d_stream)
        self._memory2stream = nn.MultiheadAttention(d_stream, heads, 0.1, kdim=d_memory, vdim=d_memory)


    def forward(self,
                text_stream: torch.Tensor,
                residual: torch.Tensor,
                memory: torch.Tensor):

        #Import the residuals, and transfer summaries into memory.
        stream = self._intake_stream_norm(text_stream + residual)
        memory = self._intake_memory_norm(memory)

        memory = self._transfer_into_memory_norm(memory + self._stream2memory(memory, stream, stream))

        #process layers in parallel


class SummarizationLayer(nn.Module):
    """

    Contains a primary

    """

class EncoderSubLayer(nn.Module):
    """

    A complete encoder sequence, without additional parallel instances
    . Consists of the
    residual bypass, resampling, globalization,
    and summarization which occurs for a single layer.

    Returns the layer residuals, and the summary.

    """

    def __init__(self,
                 d_models: List[int],
                 d_ratios: List[int],
                 kernel_width: int,
                 dilation_rates: List[int],
                 supersampling: List[int],
                 dropout: float):
        """

        :param CC_Converter: The converter for going between composite and component form. Should expose d_model, d_total
        :param feedforwards: A list of feedforward layers, one per component.
        :param self_attn: A list of self_attn layers. May be banded.
        :param composite_feedforward: A feedforward layer of width d_total
        :param dropout: The aggressiveness of the dropout.
        """

        super().__init__()

        self._Converter = Architecture.CompositeComponentConverter(d_models, d_ratios, "sum")

        self._ComponentAttn = []
        self._CANorm = []
        self._ComponentFF = []
        self._CFNorm = []

        for d_model in d_models:

            attn = Learnables.BandedMultiheadedAttention(d_model,
                                                        kernel_width,
                                                        dilation_rates= dilation_rates,
                                                         supersampling=supersampling,
                                                         pad=True,
                                                         trim=True)
            self._ComponentAttn.append(attn)

            norm = nn.LayerNorm(d_model)


            self._ComponentAttn

        self._ComponentAttn = []

        self._ComponentFF = [Learnables.Feedforward(d_model, d_model*4) for d_model in d_models]


        self._TLayerNorms = nn.ModuleList([nn.LayerNorm(d_model) for d_model in CC_Converter.d_models])
        self._FLayerNorms = nn.ModuleList([nn.LayerNorm(d_model) for d_model in CC_Converter.d_models])
        self._CFLayerNorm = nn.LayerNorm(CC_Converter.d_total)



        self._Converter = CC_Converter
        self._feedforwards = nn.ModuleList(feedforwards)
        self._self_attn = nn.ModuleList(self_attn)
        self._Composite_FF = composite_feedforward

        self._dropout = nn.Dropout(dropout)

    def forward(self, tensor: torch.Tensor):
        components = self._Converter.components(tensor)
        final_components = []
        for i, (component) in enumerate(components):
            TNorm, self_attn = self._TLayerNorms[i], self._self_attn[i]
            subcomponent = self_attn(component, component, component)
            component = TNorm(component + self._dropout(subcomponent))

            FNorm, feedforward = self._FLayerNorms[i], self._feedforwards[i]
            subcomponent = feedforward(component)
            component = FNorm(component + self._dropout(subcomponent))
            final_components.append(component)

        composite = self._Converter.composite(final_components)
        composite = self._Composite_FF(composite)
        output = self._CLayerNorm(tensor + self._dropout(composite))
        return output