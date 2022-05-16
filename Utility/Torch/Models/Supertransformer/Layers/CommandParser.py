import torch
from torch import nn
from torch.nn import functional as F

from Utility.Torch.Learnables import Layers

"""

The command stream process. Responsible for
starting a particular processing iteration. Takes
in some sort of tensor stream, and last update's 
query memory residuals. 

Responsible for producing a certain quantity of the 
response tensors, and more importantly for producing
the residuals sent off to the document parser

"""

class CommandParser(nn.Module):
    """

    Accepts some sort of a string embedding.

    """
    def __init__(self, ):
