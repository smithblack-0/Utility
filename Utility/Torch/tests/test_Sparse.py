import unittest
import torch
import torch_sparse

from Utility.Torch.Sparse import Parameter

class test_SparseParameter(unittest.TestCase):
    def test_allocate(self):
        """ Test whether the initializer works at reserving memory"""

        default_parameter = Sparse.SparseParameter(13)
        custom_shape_parameter = Sparse.SparseParameter(300)
        nograd_parameter = Sparse.SparseParameter(321, requires_grad=False)

        print(default_parameter._backend.shape)
        self.assertTrue(default_parameter._backend.shape == (13,))
        self.assertTrue(custom_shape_parameter._backend.shape == (300,))
        self.assertTrue(nograd_parameter._backend.shape == (321,))

        self.assertTrue(default_parameter._backend.requires_grad)
        self.assertTrue(custom_shape_parameter._backend.requires_grad)
        self.assertFalse(nograd_parameter._backend.requires_grad)
    def test_grow(self):
        """ Test whether I can easily grow tensor parameters"""

        #Test nogrow failure
        with self.assertRaises(AssertionError):
            row = torch.empty([])
            col = torch.empty([])
            nofill = torch.empty([])
            noparam = Sparse.SparseParameter(10)
            noparam.grow_(row, col, nofill)

        #Test basic grow success
        row = torch.arange(20)
        col = torch.arange(20)
        basic_fill = torch.ones([20])

        basic_parameter_a = Sparse.SparseParameter(15) #Not enough params
        basic_parameter_b = Sparse.SparseParameter(24) #Enough params

        a_index = torch.arange(15)
        a_fill = torch.ones(15)
        b_index = torch.arange(20)
        a_expected_result = torch_sparse.SparseTensor(row=a_index, col=a_index, value=a_fill)
        b_expected_result_1 = torch_sparse.SparseTensor(row=b_index, col=b_index, value=basic_fill)

        with self.assertWarns(RuntimeWarning):
            basic_parameter_a.grow_(row, col, basic_fill)
        basic_parameter_b.grow_(row, col, basic_fill, False)

        assert torch.all(basic_parameter_a._backend == torch.ones([15])), "backend not copied in correctly"
        assert torch.all(basic_parameter_b._backend[:15] == torch.ones([15])), "backend not copied corretly"

        assert a_expected_result == basic_parameter_a.sparse, "expected and actual sparse do not match"
        assert b_expected_result_1 == basic_parameter_b.sparse, "expected and actual sparse do not match"
        #Test doublegrow success

        new_fill = 3*torch.ones([20])
        b_index = torch.arange(24)
        b_fill2 = torch.concat([basic_fill, new_fill[:4]], dim=0)
        b_expected_result_2 = torch_sparse.SparseTensor(row=b_index, col=b_index, value=b_fill2)

        with self.assertWarns(RuntimeWarning):
            basic_parameter_b.grow_(row+20, col+20, new_fill)

        print(b_expected_result_2)
        print(basic_parameter_b.sparse)
        assert basic_parameter_b.sparse == b_expected_result_2, "Doublegrowth result not what expected"
        assert torch.equal(basic_parameter_b._backend, b_fill2), "Backend not as expected"


        #Test nonsuppressed error fails when full.
        with self.assertRaises(IndexError):
            basic_parameter_a.grow_(row, col, new_fill, False)
    def test_prune(self):
        """ Tests that the pruning function works as expected"""

class test_Capture(unittest.TestCase):
    """

    Tests the ability of the capture module to catch gradient information.

    """
    def test_startup(self):







if __name__ == '__main__':
    unittest.main()