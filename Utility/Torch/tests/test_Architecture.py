import unittest
import torch
from torch import nn
from Utility.Torch import Architecture

class test_GLC(unittest.TestCase):

    def test_basic(self):

        tensor = torch.randn([10, 100, 50])
        d_models = [10, 30, 10]
        ratios = [10, 1, 0]

        class passthrough_module(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, tensor):
                tensor = tensor*True
                return tensor

        modules = [passthrough_module() for _ in d_models]
        tester = Architecture.GLCSuperprocesser(d_models, ratios, modules)
        test = tester(tensor)



