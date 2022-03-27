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

    
        
class testArchive(unittest.TestCase):
    """

    This is the test case for the
    archive class of Archivist

    """
    def test_Constructor(self):
        """ Tests the constructor and return parameters in a few different configurations"""

        record = torch.randn([30, 10, 20])
        index = torch.randn([30, 10,4])
        Learnables.Archivist.Archive(record, index)
    def test_Basic_Truth_Lookup(self):
        """ Tests whether a lookup may be performed, and if the shape is correct."""
        #create test fixtures
        record = torch.randn([30, 10, 20])
        index = torch.ones([30, 10, 4])
        query = torch.ones([30, 34, 4])
        test_fixture = Learnables.Archivist.Archive(record, index)

        expected_normal_shape = torch.tensor([30, 34, 10, 20])
        expected_training_shape = torch.tensor([30, 34, 11, 20])

        #perform tests
        normal_test = test_fixture(query)
        training_test = test_fixture(query, True)

        normal_result = torch.equal(torch.tensor(normal_test.shape), expected_normal_shape)
        training_result = torch.equal(torch.tensor(training_test.shape), expected_training_shape)

        #Run assert
        self.assertTrue(normal_result, "Nonpropogating lookup came out with wrong shape")
        self.assertTrue(training_result, "Propogating lookup came out with wrong shape")
    def test_Basic_False_Lookup(self, ):
        """ Tests behavior when everything is found to be false. Tests shutoff works"""
        # create test fixtures
        record = torch.randn([30, 10, 20])
        index = torch.ones([30, 10, 4])
        query = -torch.ones([30, 34, 4])
        test_fixture = Learnables.Archivist.Archive(record, index)

        expected_normal_shape = torch.tensor([30, 34, 0, 20])
        expected_training_shape = torch.tensor([30, 34, 1, 20])

        # perform tests
        normal_test = test_fixture(query)
        training_test = test_fixture(query, True)

        normal_result = torch.equal(torch.tensor(normal_test.shape), expected_normal_shape)
        training_result = torch.equal(torch.tensor(training_test.shape), expected_training_shape)

        print(normal_test.shape)
        print(training_test.shape)
        # Run assert
        self.assertTrue(normal_result, "Nonpropogating lookup came out with wrong shape")
        self.assertTrue(training_result, "Propogating lookup came out with wrong shape")

    def test_logic(self):
        """ Test whether the actual underlying logic words. """
        # create test fixtures
        record = torch.arange(12)
        record = record.view(12, 1)

        index = torch.concat([torch.ones([6, 1]), -torch.ones([6, 1])], dim=0)

        query = torch.ones([2])
        query[1::2] = -query[1::2]
        query = query.view(2, 1)

        expected = torch.stack([torch.arange(6), torch.arange(6, 12)], dim=0).type(torch.float32)

        test_feature = Learnables.Archivist.Archive(record, index)
        #Run tests

        test_output = test_feature(query)
        test_bool = torch.equal(test_output.squeeze(), expected)
        self.assertTrue(test_bool, "Logic failed")