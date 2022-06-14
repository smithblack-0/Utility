import unittest
import torch
from torch import nn
from Utility.Torch.Archive import Architecture


class test_GLCA(unittest.TestCase):

    def test_basic(self):
        """ Tests whether a component transform, followed by a composite transform, actually works"""
        tensor = torch.randn([10, 100, 50])
        d_models = [10, 30, 10]
        ratios = [10, 1, 0]
        expected_lengths = [100, 10, 1]

        instance = Architecture.CompositeComponentConverter(d_models, ratios)

        test = instance.component(tensor)
        for item, length in zip(test, expected_lengths):
            self.assertTrue(item.shape[-2] == length)
        test = instance.composite(test)
        self.assertTrue(tensor.shape == test.shape)

    def test_jit(self):
        """ Tests whether this will jit compile"""
        tensor = torch.randn([10, 100, 50])
        d_models = [10, 30, 10]
        ratios = [10, 1, 0]

        class tester(nn.Module):
            def __init__(self):
                super().__init__()
                self.tester = Architecture.CompositeComponentConverter(d_models, ratios)
            def forward(self, tensor):
                test = self.tester.component(tensor)
                test = self.tester.composite(test)
                return test

        instance = tester()
        instance = torch.jit.script(instance)
        test = instance(tensor)
        self.assertTrue(tensor.shape == test.shape)


