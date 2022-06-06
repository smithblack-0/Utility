"""


ANAC Memory

Activity Normalized Auto Calibrating Memory consists
of several mechanisms working in tandem.

- TASK SELECTION

Task Selection consists of a process by which a series of relevant facts are returned
for a particular situation. Let a series of queries E exist. Learnable key matrix K
and value matrix V exist, and have attention performed.

- TASK NORMALIZATION:

The activity for the various memory slots is tracked. More active slots recieve
less urgent updates. Infrequently accessed locations, meanwhile, recieve large updates
when they are found to be useful.

- RUNNING CALIBRATION

Tasks for which the query and key are very similar are considered to be
alike. The most alike task has a proportion of the query inserted into the
similar slot on the key, as a mean across all examples and across all batches.

This means as query shape changes, to some degree the layer can follow along
without being trained to maximize information, simply by asking what the most
beneficial information for the moment would be.

"""
from typing import List, Optional
from Utility.Torch.Learnables import Layers

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed.optim import _FunctionalAdam



class ParameterInjectionUnit(nn.Module):
    """
    Parameter Injection Unit.

    Parameter Memory are large blocks of parameters
    which are compatible with an embedded stream
    as though they are embeddings themselves.

    The process of Parameter Injection is a process
    of conditionally injecting whole blocks of parameters,
    into a running embedded stream as though it were an
    embedding itself. Two tasks exist. First, the module
    must figure out what parameter block to inject, and
    when. Second, the module must train the parameter
    blocks to provide useful context.

    Generally, a high head count and softmax mode are
    desirable. The mem width should provide some reasonable
    choices, such as 10. Should a high head count not
    be feasable, an alternative would be a low head count
    with a much larger memory. Be aware, however, this
    will increase the number of parameters in proportion.
    """
    def __init__(self,
                 embedding_width: int,
                 mem_width: int,
                 heads: int,
                 mode: str = "softmax",
                 ):
        super().__init__()

        assert embedding_width % heads == 0
        self.mode = mode

        head_channel_width = embedding_width//heads
        key = torch.zeros([heads, mem_width, head_channel_width])
        value = torch.zeros([heads, mem_width, head_channel_width])

        nn.init.kaiming_uniform_(key)
        nn.init.kaiming_uniform_(value)

        self.QueryProj = Layers.Linear(embedding_width, [heads, head_channel_width])
        self.Key = nn.Parameter(key)
        self.Value = nn.Parameter(value)
        self.Dropout = nn.Dropout(dropout)
        self.DeheadProj = Layers.Linear([heads, head_channel_width], embedding_width)

    def forward(self, tensor: torch.Tensor):

        #Get key, value, and query prepped

        query = self.QueryProj(tensor).transpose(-2, -3) #(..., head, items,  head_embedding)
        key = self.Key #(..., head, mem, head_embedding)
        value = self.Value #(..., head, mem, head_embedding)

        #Perform scoring and attention

        logits = torch.matmul(query, key.transpose(-1, -2)) # (... head, items, mem)

        if self.mode == "softmax":
            score = torch.softmax(logits, dim=-1) #(..., head, items, mem)
        elif self.mode == "sigmoid":
            score = torch.sigmoid(logits) #(..., head, items, mem)
        else:
            raise ValueError("Invalid mode specified")

        attn = torch.matmul(score, value)/torch.sqrt(torch.tensor(query.shape[-1])) #(..., head, items, mem)
        output = self.DeheadProj(output.transpose(-2, -3))

        return output
