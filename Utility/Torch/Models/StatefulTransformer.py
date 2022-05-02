

import torch
from torch import nn
from Utility.Torch import Architecture
from Utility.Torch import Learnables



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
    Consists of:
    LAyernorm plus add:
        Component Decompose
            LayerNorm plus Add:
                PseudoBanded
    
    
    """
    def __init__(self,
                 CC_Converter: nn.Module
                 feedforwards: List[nn.Module],
                 transformers: List[nn.Module],
                 composite_feedforward: nn.Module
                 dropout: float):
        
        self._TLayerNorms = nn.ModuleList([nn.LayerNorm(d_model) for d_model in CC_Converter.d_models])
        self._FLayerNorms = nn.ModuleList([nn.LayerNorm(d_model) for d_model in CC_Converter.d_models])
        self._CLayerNorm = nn.LayerNorm(CC_Converter.d_total)
      
        self._Converter = CC_Converter
        self._feedforwards = nn.ModuleList(feedforwards)
        self._transformers = nn.ModuleList(transformers)
        self._composite_feedforwad = composite_feedforward
    
        
    def forward(self, tensor: torch.Tensor):
        
        
        
        
        
