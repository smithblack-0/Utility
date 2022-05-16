import unittest
from typing import List

import torch
from torch import nn
from Utility.Torch.Models.Supertransformer.Layers import EnsembleTools as Ensemble



class test_Submodel(unittest.TestCase):
    """
    Tester for the submodel unit of the superensemble logic.
    """
    class sublayer_mockup(nn.Module):
        """
        Just returns what it was given
        """
        def __init__(self):
            super().__init__()
        def forward(self, stream, memory):
            return stream, memory

    def test_basic(self):
        """ Tests whether mockup feedthrough works. """
        tensor = torch.randn([4, 10])
        memory = torch.randn([2, 20])

        sublayers1 = [self.sublayer_mockup() for _ in range(5)]
        sublayers2 = [self.sublayer_mockup() for _ in range(5)]

        submodel1 = Ensemble.SubModel(sublayers1)
        submodel2 = Ensemble.SubModel(sublayers2)


        tensor, memory, tensor_residuals, memory_residuals = submodel1(tensor, memory)
        tensor, memory, tensor_residuals, memory_residuals = submodel2(tensor, memory, tensor_residuals, memory_residuals)
    def test_torchscript_compiles(self):
        """ Tests whether or not the model compiles in torchscript"""
        sublayers = [self.sublayer_mockup() for _ in range(5)]
        model = Ensemble.SubModel(sublayers)
        model = torch.jit.script(model)

        tensor = torch.randn([4, 10])
        memory = torch.randn([2, 20])

        tensor, memory, tensor_residuals, memory_residuals = model(tensor, memory)
        tensor, memory, tensor_residuals, memory_residuals = model(tensor, memory, tensor_residuals,
                                                                       memory_residuals)

class test_SuperEnsemble(unittest.TestCase):
    """

    Tests the ability of the SuperEnsemble class to
    run SuperEnsemble as a whole.

    """
    class sublayer_mockup(nn.Module):
        """
        Just returns what it was given
        """
        def __init__(self):
            super().__init__()
        def forward(self, stream, memory):
            return stream, memory
    def test_basics(self):
        """ Test the ability to feed data through the model"""

        tensor = torch.randn([4, 10])

        sublayers = [self.sublayer_mockup() for _ in range(5)]

        submodel1 = Ensemble.SubModel(sublayers)
        submodel2 = Ensemble.SubModel(sublayers)
        submodels = [submodel1, submodel2]

        mem1 = Ensemble.MemSeed([20], torch.float32)
        mem2 = Ensemble.MemSeed([20], torch.float32)
        mems = [mem1, mem2]

        test_augments = [item() for item in mems]

        model = Ensemble.SuperEnsemble(submodels, mems)

        (tensor_output, memory_outputs), (tensor_res, task_res) = model(tensor)
        (tensor_output, memory_outputs), (tensor_res, task_res) = model(tensor, tensor_res, task_res)
        (tensor_output, memory_outputs), (tensor_res, task_res) = model(tensor, tensor_res, task_res, test_augments)
    def test_torchscript_compile(self):
        """ Test the ability to compile, under torchscript"""
        tensor = torch.randn([4, 10])

        sublayers = [self.sublayer_mockup() for _ in range(5)]

        submodel1 = Ensemble.SubModel(sublayers)
        submodel2 = Ensemble.SubModel(sublayers)
        submodels = [submodel1, submodel2]

        mem1 = Ensemble.MemSeed([20], torch.float32)
        mem2 = Ensemble.MemSeed([20], torch.float32)
        mems = [mem1, mem2]

        model = Ensemble.SuperEnsemble(submodels, mems)
        model = torch.jit.script(model)
        model(tensor)



