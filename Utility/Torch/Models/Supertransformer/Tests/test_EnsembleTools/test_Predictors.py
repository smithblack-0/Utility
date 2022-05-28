import torch
import unittest
from torch import nn
from Utility.Torch.Models.Supertransformer.EnsembleTools import Scheduler
from Utility.Torch.Models.Supertransformer.EnsembleTools import Teardown
from Utility.Torch.Models.Supertransformer import StreamTools

class test_CatagoricalPredictor(unittest.TestCase):
    """
    Test case for the catagorical predictor.
    """