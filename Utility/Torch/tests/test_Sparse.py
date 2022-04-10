import unittest
import torch
from Utility.Torch.Sparse import Sparse

class test_SparseParameter(unittest.TestCase):
    def test_allocate(self):
        """ Test whether the initializer works at reserving memory"""

        default_parameter = Sparse.SparseParameter(13)
        custom_shape_parameter = Sparse.SparseParameter(300, (4, 4))
        nograd_parameter = Sparse.SparseParameter(321, requires_grad=False)


        self.assertTrue(default_parameter.backend.shape == (13, 1))
        self.assertTrue(custom_shape_parameter.backend.shape == (300, 4, 4))
        self.assertTrue(nograd_parameter.backend.shape == (321, 1))

        self.assertTrue(default_parameter.backend.requires_grad)
        self.assertTrue(custom_shape_parameter.backend.requires_grad)
        self.assertFalse(nograd_parameter.backend.requires_grad)
    def test_grow(self):
        """ Test whether I can easily grow tensor parameters"""

        basic_parameter = Sparse.SparseParameter(15, (3,))
        advanced_parameter = Sparse.SparseParameter(34, (4,5))


        #Test basic grow
        row = torch.arange(20)
        col = torch.arange(20)
        basic_fill = torch.ones([20])
        advanced_fill = torch.ones([20, 4, 5])

        basic_parameter.grow_(row, col, basic_fill)
        def fail_grow():
            basic_parameter.grow_(row, col, basic_fill, False)
        self.assertRaises(fail_grow, )
        print(basic_parameter.sparse)



if __name__ == '__main__':
    unittest.main()