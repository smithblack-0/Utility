import unittest
import torch

from Utility.Torch.Rearrange import view
from Utility.Torch.Rearrange import local


class testLocal(unittest.TestCase):
    def testAsLayer(self):
        """
        Test if a simple layer works.
        """

        # Perform direct logic test
        tensor = torch.arange(30)
        kernel, stride, dilation = 1, 1, 1
        final = tensor.unsqueeze(-1)

        test = local(tensor, kernel, stride, dilation)
        test = torch.all(test == final)

        self.assertTrue(test, "Logical failure: results did not match manual calculation")
    def testKernel(self):
        """
        Test if a straightforward local kernel, as used in a convolution, works
        """

        # Perform kernel compile and logical test
        tensor = torch.tensor([0, 1, 2, 3, 4, 5])
        final = torch.tensor([[0 ,1] ,[1, 2], [2, 3], [3, 4], [4, 5]])
        kernel, stride, dilation = 2, 1, 1

        test = local(tensor, kernel, stride, dilation)
        test = torch.all(test == final)
        self.assertTrue(test, "Logical failure: Kernels not equal")
    def testStriding(self):
        """
        Test if a strided kernel, as used in a convolution, works
        """

        # Perform striding compile and logical test
        tensor = torch.tensor([0, 1, 2, 3, 4, 5])
        final = torch.tensor([[0], [2], [4]])
        kernel, stride, dilation = 1, 2, 1

        test = local(tensor, kernel, stride, dilation)
        test = torch.all(test == final)
        self.assertTrue(test, "Logical failure: striding did not match")
    def testDilation(self):
        """
        Test if a combination of dilated kernels works.
        """

        # Perform dilation test
        tensor = torch.tensor([0, 1, 2, 3, 4, 5])
        final = torch.tensor([[0, 2], [1, 3], [2, 4], [3, 5]])
        final2 = torch.tensor([[0, 2, 4], [1, 3, 5]])
        final3 = torch.tensor([[0, 3] ,[1 ,4] ,[2 ,5]])

        kernel1, stride1, dilation1 = 2, 1, 2
        kernel2, stride2, dilation2 = 3, 1, 2
        kernel3, stride3, dilation3 = 2, 1, 3

        test = local(tensor, kernel1, stride1, dilation1)
        test2 = local(tensor, kernel2, stride2, dilation2)
        test3 = local(tensor, kernel3, stride3, dilation3)

        test = torch.all(final == test)
        test2 = torch.all(final2 == test2)
        test3 = torch.all(final3 == test3)

        self.assertTrue(test, "Logical failure: dilation with kernel did not match")
        self.assertTrue(test2, "Logical failure: dilation with kernel did not match")
        self.assertTrue(test3, "Logical failure: dilation with kernel did not match")
    def testRearranged(self):
        """
        Test if a tensor currently being viewed, such as produced by swapdims, works
        """
        # make tensor
        tensor = torch.arange(20)
        tensor = tensor.view((2, 10))  # This is what the final buffer should be viewed with respect to
        tensor = tensor.swapdims(-1, -2).clone()  # Now a new tensor with a new buffert
        tensor = tensor.swapdims(-1, -2)  # Buffer is being viewed by stridings. This could fuck things up

        # Declare kernal, striding, final
        kernel, striding, dilation = 2, 2, 2

        # Make expected final
        final = []
        final.append([[0 ,2] ,[2 ,4], [4 ,6] ,[6 ,8]])
        final.append([[10, 12] ,[12, 14] ,[14, 16], [16, 18]])
        final = torch.tensor(final)

        # test
        test = local(tensor, kernel, striding, dilation)
        test = torch.all(final == test)
        self.assertTrue(test, "Logical failure: buffer issues")



