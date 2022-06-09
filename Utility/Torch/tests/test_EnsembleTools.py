"""

Test the content of the
ensemble tools module

"""

import unittest
import torch
from torch import nn
from torch.nn import functional as F

import Utility.Torch.EnsembleTools.submodel
from Utility.Torch import EnsembleTools


class test_AdditiveSubmodel(unittest.TestCase):
    """
    Test the submodel relying solely on addition

    """
    def test_basic(self):
        """ Tests that a additive submodel runs at all"""
        embedding_width = 10
        layers = [nn.Linear(embedding_width, embedding_width) for _ in range(5)]
        submodel = Utility.Torch.EnsembleTools.submodel.AdditiveSubmodel(embedding_width, 0.1, layers)

        tensor = torch.randn([30, embedding_width])
        output, residual = submodel(tensor, None)
        output, residual = submodel(tensor, residual)
    def test_torchscript(self):
        """ Tests that an additive submodel is torchscript compatible."""
        embedding_width = 10
        layers = [nn.Linear(embedding_width, embedding_width) for _ in range(5)]
        submodel = Utility.Torch.EnsembleTools.submodel.AdditiveSubmodel(embedding_width, 0.1, layers)
        submodel = torch.jit.script(submodel)

        tensor = torch.randn([30, embedding_width])
        output, residual = submodel(tensor, None)
        output, residual = submodel(tensor, residual)

class test_ConcatSubmodel(unittest.TestCase):
    """
    tests that the concative submodel is working properly.
    """
    def test_basic(self):
        """ tests that a basic case is running. """
        embedding_width = 10
        layers = [nn.Linear(embedding_width, embedding_width) for _ in range(5)]
        submodel = Utility.Torch.EnsembleTools.submodel.ConcativeSubmodel(embedding_width, 0.1, layers)

        tensor = torch.randn([30, embedding_width])
        output, residual = submodel(tensor, None)
        output, residual = submodel(tensor, residual)
    def test_torchscript(self):
        """ Test that torchscript compiles okay"""
        embedding_width = 10
        layers = [nn.Linear(embedding_width, embedding_width) for _ in range(5)]
        submodel = Utility.Torch.EnsembleTools.submodel.ConcativeSubmodel(embedding_width, 0.1, layers)
        submodel = torch.jit.script(submodel)

        tensor = torch.randn([30, embedding_width])
        output, residual = submodel(tensor, None)
        output, residual = submodel(tensor, residual)

class test_GateSubmodel(unittest.TestCase):
    """
    Tests whether the gate submodel runs
    """
    def test_basic(self):
        """ tests whether anything works"""
        embedding_width = 10
        layers = [nn.Linear(embedding_width, embedding_width) for _ in range(5)]
        submodel = Utility.Torch.EnsembleTools.submodel.GateSubmodel(embedding_width, 0.1, layers)

        tensor = torch.randn([30, embedding_width])
        output, residual = submodel(tensor, None)
        output, residual = submodel(tensor, residual)
    def test_torchscript(self):
        """ test whether torchscrip compiles"""
        embedding_width = 10
        layers = [nn.Linear(embedding_width, embedding_width) for _ in range(5)]
        submodel = Utility.Torch.EnsembleTools.submodel.GateSubmodel(embedding_width, 0.1, layers)
        submodel = torch.jit.script(submodel)

        tensor = torch.randn([30, embedding_width])
        output, residual = submodel(tensor, None)
        output, residual = submodel(tensor, residual)

class test_crossentropyboost(unittest.TestCase):
    def test_basic(self):
        test_vector = torch.randn([10, 20, 4, 20])
        test_labels = torch.randint(20, [10, 4])
        loss_func = EnsembleTools.CrossEntropyBoost(20, 1, 0.1, 0.1)
        loss = loss_func(test_vector, test_labels)

