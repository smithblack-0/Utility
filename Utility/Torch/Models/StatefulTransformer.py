

import torch
from torch import nn
from Utility.Torch import Architecture
from Utility.Torch import Learnables

from typing import List


class Encoder():
    """ Encodes an incoming command or document"""
    pass

class State():
    """Holds and updates the state document backend as questions are made"""
    pass

class Decoder():
    """Decodes a response from the provided state"""
    
class EncoderLayer(nn.Module):
    """

    A complete EncoderLayer.

    Performs component breakdown and processing,
    then recombines result and performs holistic
    processing.

    Add plus layernorm for every step
    
    """
    def __init__(self,
                 CC_Converter: nn.Module,
                 feedforwards: List[nn.Module],
                 self_attn: List[nn.Module],
                 composite_feedforward: nn.Module,
                 dropout: float):
        """

        :param CC_Converter: The converter for going between composite and component form. Should expose d_model, d_total
        :param feedforwards: A list of feedforward layers, one per component.
        :param self_attn: A list of self_attn layers. May be banded.
        :param composite_feedforward: A feedforward layer of width d_total
        :param dropout: The aggressiveness of the dropout.
        """

        super().__init__()
        
        self._TLayerNorms = nn.ModuleList([nn.LayerNorm(d_model) for d_model in CC_Converter.d_models])
        self._FLayerNorms = nn.ModuleList([nn.LayerNorm(d_model) for d_model in CC_Converter.d_models])
        self._CLayerNorm = nn.LayerNorm(CC_Converter.d_total)
      
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

class CommandLayer(nn.Module):
    """

    The stateful updating layer.

    Accepts an existing worldstate, and a
    embedding representing a command of some sort. It
    then updates the worldstate based on the contents of the
    command.

    The update process operates on components for efficiency, but
    does not use banding. This means the update rate is proportional
    to N*L, where L is the length of the update. It is thus
    recommended to keep update lengths low.

    THIS is what is going to allow rapid, stateful updating. For example,
    redefining words: "For every word potato you would say in the future,
    replace it with strawberry."

    """

    def __init__(self,
                CC_Converter: nn.Module,
                feedforwards: List[nn.Module],
                self_attn: List[nn.Module],
                update_attn: List[nn.Module],
                composite_feedforward: nn.Module,
                dropout: float):
        """


        :param CC_Converter: A converter capable of transforming between component and composite format. Should also
            display d_model and d_total
        :param feedforwards:
            A list of feedforward layers, one per component. d_models in same sequence as CC_Converter d_model
        :param self_attn:
            A list of self attention layers, one per component. May be banded
        :param update_attn:
            A list of update layers, one per component. May NOT be banded.
        :param composite_feedforward:
            A single feedforward layer of width d_total.
        :param dropout:
            How aggressive the dropout should be.
        """
        super().__init__()

        self._SLayerNorms = nn.ModuleList([nn.LayerNorm(d_model) for d_model in CC_Converter.d_models])
        self._ULayerNorms = nn.ModuleList([nn.LayerNorm(d_model) for d_model in CC_Converter.d_models])
        self._FLayerNorms = nn.ModuleList([nn.LayerNorm(d_model) for d_model in CC_Converter.d_models])
        self._CLayerNorm = nn.LayerNorm(CC_Converter.d_total)

        self._Converter = CC_Converter
        self._feedforwards = nn.ModuleList(feedforwards)
        self._self_attn = nn.ModuleList(self_attn)
        self._update_attn = nn.ModuleList(update_attn)
        self._Composite_FF = composite_feedforward

        self._dropout = nn.Dropout(dropout)

    def forward(self, worldstate: torch.Tensor, update: torch.Tensor):


        components = self._Converter.components(worldstate)
        update = self._Converter.split(update)
        final_components = []
        for i, (component) in enumerate(components):

            #Self attention
            SNorm, self_attn = self._SLayerNorms[i], self._self_attn[i]
            subcomponent = self_attn(component, component, component)
            component = SNorm(component + self._dropout(subcomponent))

            #Update attention
            UNorm, update_attn, component_update = self._ULayerNorms[i], self._update_attn[i], update[i]
            subcomponent = update_attn(component, component_update, component_update)
            component = UNorm(component + self._dropout(subcomponent))

            #Feedforward
            FNorm, feedforward = self._FLayerNorms[i], self._feedforwards[i]
            subcomponent = feedforward(component)
            component = FNorm(component + self._dropout(subcomponent))

            final_components.append(component)

        #Composite feedforward.
        composite = self._Converter.composite(final_components)
        composite = self._Composite_FF(composite)
        output = self._CLayerNorm(worldstate + self._dropoout(composite))
        return output

class DecoderLayer(nn.Module):
    """

    Decoder layer.

    Utilized to actually decode a worldstate,
    presumably in response to a command.

    Called with a particular worldstate, and the
    current autorecursive state. It considers the
    entire space of possibilities

    """

    def __init__(self,
                d_model: int,
                feedforward: nn.Module,
                self_attn: nn.Module,
                queried_attn: nn.Module,
                dropout: float):
        """


        :param d_model: The total width of the incoming embeddings
        :param feedforward: A feedforward layer
        :param self_attn: A self_attn layer. Banded. Width d_model
        :param queried_attn: A queried attn layer. Not Banded. Width d_model.
        :param dropout: The dropout aggressiveness.
        """

        super().__init__()

        self._self_attn = self_attn
        self._snorm = nn.LayerNorm(d_model)

        self._query_attn = queried_attn
        self._qnorm = nn.LayerNorm(d_model)

        self._feedforward = feedforward
        self._fnorm = nn.LayerNorm(d_model)

        self._dropout = nn.Dropout(dropout)
    def forward(self,
                worldstate: torch.Tensor,
                stream: torch.Tensor,
                mask: torch.Tensor):

        #Self attn.
        substream = self._self._attn(stream, stream, stream)
        stream = self._snorm(stream + self._dropout(substream))

        #Attn.
        substream = self._query_attn(stream, worldstate, worldstate, mask)
        stream = self._qnorm(stream + self._dropout(substream))

        #Feedforward.
        substream = self._feedforward(stream)
        stream = self._fnorm(stream + self._dropout(substream))
        return stream








        
        
        
        
