import unittest
import torch

from Utility.Torch import Learnables

class testLinear(unittest.TestCase):
    """
    This is the test feature for the linear layer.
    """
    def test_Regular(self):
        """ Tests if the standard pytorch linear layer is reproduced"""

        tensor = torch.rand([33, 2, 5])
        tester = Learnables.Linear(5, 10)
        test = tester(tensor)
        self.assertTrue(test.shape[-1] == 10, "Regular pytorch layer not reproduced")
    def test_Reshapes(self):
        """ Tests whether the reshape functionality is working in isolation """
        #Define test tensor
        tensor = torch.rand([30, 20, 15])

        #Define test layers
        test_expansion = Learnables.Linear(15, (5, 3))
        test_collapse = Learnables.Linear((20, 15), 300)
        test_both = Learnables.Linear((20, 15), (10, 30))

        #Perform tests

        test_expansion_result = test_expansion(tensor)
        test_collapse_result = test_collapse(tensor)
        test_both_result = test_both(tensor)

        expansion_bool = [*test_expansion_result.shape] == [30, 20, 5, 3]
        collapse_bool = [*test_collapse_result.shape] == [30, 300]
        both_bool = [*test_both_result.shape] == [30, 10, 30]

        #Assert results
        self.assertTrue(expansion_bool, "Reshape: Expansion failed")
        self.assertTrue(collapse_bool, "Reshape: collapse failed")
        self.assertTrue(both_bool, "Reshape: Compound failed")
    def test_Heading(self):
        """ Tests whether the head kernels and bias are implemented such that calling works"""

        tensor = torch.randn([10, 30, 20, 10])

        #Create test layers

        test_single = Learnables.Linear(10, 20, 20)
        test_multiple = Learnables.Linear(10, 20, (30, 20))

        #Run tests

        test_single_result = test_single(tensor)
        test_multiple_result = test_multiple(tensor)

    def test_Head_Independence(self):
        """ Tests whether each head is completely independent"""

