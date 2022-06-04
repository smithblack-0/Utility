"""

A module dedicated to analyzing the parameter
injection concept.

"""

import torch
from torch import nn
from Utility.Torch.Learnables import ContextTools



def construct_mock_data():
    inputs = torch.randn([512, 1, 512])
    all_labels = torch.arange(0, 512, dtype=torch.long)

def construct_model_defaults():
    defaults = {}
    defaults['embedding_width'] = 512
    defaults['memory_width'] = 4
    defaults['heads'] = 4
    defaults['integration_lr'] = 0.001
    defaults['decay_rate'] = 0.999
    defaults['dropout'] = 0.9
    return defaults

class AutoCalibrateModel(nn.Module):
    def __init__(self, defaults):
        super().__init__()

        embed_width = defaults['embedding_width']

        self.norm = nn.LayerNorm(embed_width)
        self.layer = ContextTools.AutoCalibrationInjector(**defaults)
        self.prediction = nn.Linear(embed_width, embed_width)
    def forward(self, tensor: torch.Tensor):
        tensor = self.norm(tensor)
        tensor = self.layer(tensor)
        tensor = self.prediction(tensor)
        tensor = tensor.mean(dim=-2)
        return tensor

class train()