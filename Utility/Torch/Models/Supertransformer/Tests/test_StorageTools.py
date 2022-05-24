import unittest
from typing import List

import torch
from torch import nn
from Utility.Torch.Models.Supertransformer.Layers import EnsembleTools as Ensemble
from Utility.Torch.Models.Supertransformer import StorageTools


class test_DictTensorStorage(unittest.TestCase):
    def test_basics(self):
        test_item = {'item1' : torch.randn(10), 'item2' : torch.randn(10)}
        test_instance = StorageTools.DictTensorStorage(test_item)
        test_instance['item1']
        test_dict = dict(test_instance)
        print(test_dict)
    def test_torchscript(self):
        test_item = {'item1' : torch.randn(10), 'item2' : torch.randn(10)}
        test_instance = StorageTools.DictTensorStorage(test_item)
        test_instance = torch.jit.script(test_instance)
        test_instance['item1']



