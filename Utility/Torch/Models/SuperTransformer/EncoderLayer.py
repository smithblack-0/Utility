"""

Module for keeping the encoder layer in

"""

import torch
from torch import nn
from torch.nn import functional as F




class EncoderLayer(nn.Module):
    """
    The stack consists of:

    LBSA
    GLA
    PISU
    MHEE
    FF
    """