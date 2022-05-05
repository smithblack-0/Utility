"""

An encoder sequence is used to produce a sequence of ratios
with respect to the original input length consisting of
a_0, a_1, a_2, etc, where a_0 is equal to different lengths,
and a_f is global


"""

import torch
from torch import nn
from Utility.Torch import Learnables
from Utility.Torch import Architecture


class EncoderLayer(nn.Module):
    """

    A complete EncoderLayer.

    Performs component breakdown and processing,
    then recombines result and performs holistic
    processing.

    Add plus layernorm for every step

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
        self


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