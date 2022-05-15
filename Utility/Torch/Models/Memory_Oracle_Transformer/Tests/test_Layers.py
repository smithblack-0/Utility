import unittest
from typing import List

import torch
from torch import nn
from Utility.Torch.Models.Memory_Oracle_Transformer.Layers import TextStream



class testResidualMemoryProcessing(unittest.TestCase):
    def test_build(self):

        class test(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, stream, memory, args: List[torch.Tensor]):
                return stream, memory

        item = Layers.Submodel(3, 5, [[test()]], torch.float32)
        item(torch.randn(10), [])
        item = torch.jit.script(item)