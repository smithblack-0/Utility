import unittest
import torch

from Utility.Torch import Glimpses


class testLocal(unittest.TestCase):
    def testAsLayer(self):
        """
        Test if a simple layer works.
        """

        # Perform direct logic test
        tensor = torch.arange(30)
        kernel, stride, dilation = 1, 1, 1
        final = tensor.unsqueeze(-1)

        test = Glimpses.local(tensor, kernel, stride, dilation)
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

        test = Glimpses.local(tensor, kernel, stride, dilation)
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

        test = Glimpses.local(tensor, kernel, stride, dilation)
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

        test = Glimpses.local(tensor, kernel1, stride1, dilation1)
        test2 = Glimpses.local(tensor, kernel2, stride2, dilation2)
        test3 = Glimpses.local(tensor, kernel3, stride3, dilation3)

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
        tensor = tensor.swapdims(-1, -2).clone()  # Now a new tensor with a new data buffer
        tensor = tensor.swapdims(-1, -2)  # Buffer is being viewed by stridings. This could fuck things up

        # Declare kernel, striding, final
        kernel, striding, dilation = 2, 2, 2

        # Make expected final
        final = []
        final.append([[0 ,2] ,[2 ,4], [4 ,6] ,[6 ,8]])
        final.append([[10, 12] ,[12, 14] ,[14, 16], [16, 18]])
        final = torch.tensor(final)

        # test
        test = Glimpses.local(tensor, kernel, striding, dilation)
        test = torch.all(final == test)
        self.assertTrue(test, "Logical failure: buffer issues")


class testCompressDecompress(unittest.TestCase):
    """
    This is the test suite for the compress-decompress function
    """
    def testNoCompression(self):
        """ Tests whether CompressDecompress works correctly when no compression occurs"""

        #Setup test fixtures
        tensor = torch.randn([10, 20, 30])
        compress, decompress =  Glimpses.compress_decompress(0)

        #perform test
        compressed_tensor = compress(tensor)
        expanded_tensor = decompress(compressed_tensor)
        bool_result = torch.all(tensor == expanded_tensor)

        shape_result = torch.equal(torch.tensor(compressed_tensor.shape), torch.tensor([1, 10, 20, 30]))
        #Assert result
        self.assertTrue(shape_result, "Compression occurred when none expected")
        self.assertTrue(bool_result, "No compression case failed")
    def testBasicCompression(self):
        """ Tests whether CompressDecompress works with basic compression """

        #Setup test fixtures
        tensor = torch.randn([3, 4, 20, 230,4])
        compress, decompress = Glimpses.compress_decompress(-2)

        #Perform test
        compressed_shape = torch.tensor([3*4*20, 230, 4])

        compressed_tensor = compress(tensor)
        decompressed_tensor = decompress(compressed_tensor)

        same_bool = torch.equal(decompressed_tensor, tensor)
        shape_bool = torch.equal(compressed_shape, torch.tensor(compressed_tensor.shape))

        #Assert results

        self.assertTrue(same_bool, "Basic compression failed - not the same")
        self.assertTrue(shape_bool, "Basic compression failed - compressed shape not correct")
    def testAsymmetricCompression(self):
        """ Tests whether CompressDecompress can handle changing dimensions """

        #Setup test fixtures
        tensor = torch.randn([10, 20, 3, 5])
        compress, decompress = Glimpses.compress_decompress(-2)

        #Run compressions

        compressed_a = compress(tensor)
        compressed_b = compress(tensor)

        #Modify dimensions
        compressed_a = compressed_a[..., 0]
        compressed_b = torch.stack([compressed_b, compressed_b], dim=-1)

        #Run decompressions

        decompressed_b = decompress(compressed_b)
        decompressed_a = decompress(compressed_a)