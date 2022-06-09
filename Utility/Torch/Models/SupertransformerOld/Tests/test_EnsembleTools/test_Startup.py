"""

This is a collection of tests to test out the startup layers of
ensembletools

"""
from collections import namedtuple

import torch
import unittest
from torch import nn
from Utility.Torch.Models.SupertransformerOld.EnsembleTools import Startup
from Utility.Torch.Models.SupertransformerOld import StreamTools

class testAbstract(unittest.TestCase):
    """
    Test the abstract cases have no issues
    """
    def test(self):
        Startup.AbstractEnsembleStartup()

class testReductive(unittest.TestCase):
    """
    Tests the reductive startup case.
    """
    def make_input(self)-> StreamTools.StreamTensor:
        # Generate input.
        batch_part_a_name = "channela"
        batch_part_b_name = "channelb"

        input_items = {}
        input_items[batch_part_a_name] = torch.randn([10, 20, 5, 4])
        input_items[batch_part_b_name] = torch.randn([10, 2, 5])

        losses = {}
        losses['lossa'] = torch.tensor(34.9)

        metrics = {}
        metrics['metric'] = [torch.randn(10), torch.randn(5)]
        return StreamTools.StreamTensor(input_items, losses, metrics)
    def make_ensemble(self):
        return self.make_input()
    def make_seeds(self):
        seeds = {}
        seeds['channela'] = torch.randn([4])
        return StreamTools.StreamTensor(seeds)
    def make_recursion(self):
        recursion = {}
        recursion['recursion_channel'] = torch.randn([10, 3])
        return StreamTools.StreamTensor(recursion)
    def test_constructor(self):
        Startup.ReductionStartup(self.make_seeds(), mode='sum')
    def test_basic(self):
        seeds = self.make_seeds()
        instance = Startup.ReductionStartup(seeds)
        input = self.make_input()
        output = instance(input, None, None, None)

        expecteda = input['channela'] + seeds['channela']
        self.assertTrue(torch.all(expecteda == output['channela']))
    def test_complex(self):
        seeds = self.make_seeds()
        input = self.make_input()
        recursive = self.make_recursion()
        ensemble = self.make_ensemble()

        instance = Startup.ReductionStartup(seeds)
        output = instance(input, ensemble, recursive)

        expecteda = input['channela'] + ensemble['channela'] + seeds['channela']
        self.assertTrue(torch.all(expecteda == output['channela']))
    def test_torchscript_compiles(self):
        seeds = self.make_seeds()
        input = self.make_input()
        recursive = self.make_recursion()
        ensemble = self.make_ensemble()
        aux_class = namedtuple('test', 'x y')
        aux = aux_class(1, 1)

        print(StreamTools._MergeHelper)
        instance = Startup.ReductionStartup(seeds)
        instance = torch.jit.script(instance)
        output = instance(input, ensemble, recursive, aux)

class testConcat(unittest.TestCase):
    """
    Tests the reductive startup case.
    """
    def make_input(self)-> StreamTools.StreamTensor:
        # Generate input.
        batch_part_a_name = "channela"
        batch_part_b_name = "channelb"

        input_items = {}
        input_items[batch_part_a_name] = torch.randn([10, 20, 5, 4])
        input_items[batch_part_b_name] = torch.randn([10, 2, 5])

        losses = {}
        losses['lossa'] = torch.tensor(34.9)

        metrics = {}
        metrics['metric'] = [torch.randn(10), torch.randn(5)]
        return StreamTools.StreamTensor(input_items, losses, metrics)
    def make_ensemble(self):
        return self.make_input()
    def make_seeds(self):
        seeds = {}
        seeds['channela'] = torch.randn([4])
        return StreamTools.StreamTensor(seeds)
    def make_defaults(self):
        recursion = {}
        recursion['recursion_channel'] = torch.randn([10, 3])
        return StreamTools.StreamTensor(recursion)
    def test_constructor(self):
        Startup.ReductionStartup(self.make_seeds(), mode='sum')
    def test_basic(self):
        seeds = self.make_seeds()
        instance = Startup.ConcatStartup(seeds)
        inputitem = self.make_input()
        output = instance(inputitem, None, None, None)
    def test_complex(self):
        seeds = self.make_seeds()
        input = self.make_input()
        recursive = self.make_defaults()
        ensemble = self.make_ensemble()
        aux_class = namedtuple('test', 'x y')
        aux = aux_class(1, 1)

        instance = Startup.ConcatStartup(seeds, recursive)
        output1 = instance(input, ensemble)
        output2 = instance(input, ensemble, recursive, aux)
    def test_torchscript_compiles(self):
        seeds = self.make_seeds()
        input = self.make_input()
        recursive = self.make_defaults()
        ensemble = self.make_ensemble()

        instance = Startup.ConcatStartup(seeds, recursive)
        instance = torch.jit.script(instance)
        output1 = instance(input, ensemble)
        output2 = instance(input, ensemble, recursive)