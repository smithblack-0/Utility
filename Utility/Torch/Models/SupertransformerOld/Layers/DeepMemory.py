"""

A class for the deep memory process. Deep memory is a flavor spawned by the techniques
displayed in Memorizing Transformers (https://arxiv.org/abs/2203.08913). However,
rather thqn saving each instance to an external memory bank, instead we search a
space of differential memory, and only train the topk instances

"""
from typing import Optional

import torch
from torch import nn
from Utility.Torch.Learnables import Layers

class DeepMemoryTransformer(nn.Module):
    """
    Deep Memory is designed to allow efficient computation and collection
    of facts gathered from a variety of sources with minimal overhead.

    The input to the layer is, as is standard, the query. The key and
    value, however, are generated internally from a stored bank of
    parameters which are intended to change rapidly

    TopK is used to limit the regions which may be active at a particular
    time, providing some degree of binning
    """
    def __init__(self,
                 query_width: int,
                 output_width: int,
                 memory_length: int,
                 heads: int,
                 topk: int,
                ):
        """
        :param query_width: How wide the query embedding width is
        :param output_width: How wide the output width will be
        :param memory_length: How long the memory will be.
        :param heads: The number of heads to make.
        :param topk: The number of entities to keep per head.
        """
        assert query_width % heads == 0

        super().__init__()
        head_width = query_width//heads
        memory = torch.zeros([heads, memory_length, head_width], requires_grad=True)
        memory = torch.nn.init.kaiming_uniform(memory, requires_grad=True)

        self.memory = nn.Parameter(memory, requires_grad=True)
        self.topk = topk

        self.query_projector([query_width], [heads, head_width])
        self.key_projector([head_width], [head_width], heads)
        self.final_projector([heads, head_width], [output_width])
    def forward(self, tensor: torch.Tensor):

        query = self.query_projector(tensor).transpose(-2, -3)
        keys = self.key_projector(self.memory).unsqueeze(-1)
        score = torch.matmul(query, keys).squeeze(-1)

        top, topindices = score.topk(self.topk, dim=-2)
        score = torch.sigmoid(top) # (..., head, items, memitems)
        values = self.memory[..., topindices, :] # (..., head, memitems, encodings)
        output = torch.matmul(score, values)

        output = self.final_projector(output.transpose(-2, -3))
        return output







