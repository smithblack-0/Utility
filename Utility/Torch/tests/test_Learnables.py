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
        
        #Create tensors
        tensor_a = torch.stack([torch.zeros([20]), torch.zeros([20])])
        tensor_b = torch.stack([torch.zeros([20]), torch.ones([20])])
        
        #create tester
        
        test_head_independence = Learnables.Linear(20, 20, 2)
        
        #Run tests
        
        test_result_a = test_head_independence(tensor_a)
        test_result_b = test_head_independence(tensor_b)
        
        #Analyze and assert result
        result_bool = torch.all(test_result_a[0] == test_result_b[0])
        self.assertTrue(result_bool, "Heads were found to be interacting")
    def test_gradients(self):
        """Test whether or not gradients are propogating properly"""
        test_tensor = torch.randn([20, 10])
        
        #Develop test layer
        test_grad = Learnables.Linear((20, 10), 1)
        
        #Develop optim
        test_optim = torch.optim.SGD(test_grad.parameters(), lr=0.01)
        
        #perform test
        test_result = test_grad(test_tensor)
        test_result.backward()
        
        test_optim.step()


