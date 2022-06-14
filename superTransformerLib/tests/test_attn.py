import unittest
import torch

from superTransformerLib.transformerLib import Attention


def shape_equal(shape1, shape2):
    shape1 = torch.tensor(shape1)
    shape2 = torch.tensor(shape2)
    return torch.all(shape1 == shape2)


class test_dotproductattention(unittest.TestCase):
    """
    Tests the dot product attention function loose in the
    function module.
    """
    def test_basic_dotproductattn(self):
        """Tests if a random, no frills case works safely."""
        query = torch.randn([3, 5, 10, 20])
        content = torch.randn([3, 5, 30, 20])
        attn = Attention._dot_product_attention(query, content, content)
    def test_masked_dotproductattn(self):
        """ tests if a masked case functions correctly """
        query = torch.randn([3, 5, 10, 20])
        content = torch.randn([3, 5, 30, 20])
        mask = torch.randn([10, 30]) > 0.5
        attn = Attention._dot_product_attention(query, content, content, mask)
    def test_torchscript_compile(self):
        test = torch.jit.script(Attention._dot_product_attention)


class test_MultiHeadedAttention(unittest.TestCase):
    """
    Tests whether multiheaded attention is operating correctly.

    Tests include:

    -- does it work at all
    -- does it work when reshaping embeddings
    -- does it work when masking
    -- do ensembles work
    -- are ensembles processed independently
    -- does torchscript compile, so I can get an IR or ONNX if I need it?


    """
    def test_basic(self):
        """ test if no frills attention works"""
        query = torch.randn([3, 5, 10, 32])
        content = torch.randn([3, 5, 30, 32])
        layer = Attention.MultiHeadedAttention(32, 32, 32, 4)
        attn = layer(query, content, content)
        self.assertTrue(shape_equal([3, 5, 10, 32], attn.shape))

    def test_uneven_embeddings(self):
        """ tests if attention works when the embeddings are changing shape """
        query = torch.randn([3,5,10, 64])
        content = torch.randn([3, 5, 30, 16])
        layer = Attention.MultiHeadedAttention(64, 16, 32, 4)
        attn = layer(query, content, content)
        self.assertTrue(shape_equal([3, 5, 10, 32], attn.shape))
    def test_masked_embeddings(self):
        """ tests if attention works when a mask is involved """
        query = torch.randn([3,5,10, 64])
        content = torch.randn([3, 5, 30, 16])
        mask = torch.randn([10, 30]) > 0.5
        layer = Attention.MultiHeadedAttention(64, 16, 32, 4)
        attn = layer(query, content, content, mask)
        self.assertTrue(shape_equal([3, 5, 10, 32], attn.shape))
    def test_ensemble_capable(self):
        """ Tests whether the class functions correctly when provided an ensemble"""
        query = torch.randn([3,5,10, 64])
        content = torch.randn([3, 5, 30, 16])
        mask = torch.randn([10, 30]) > 0.5
        layer = Attention.MultiHeadedAttention(64, 16, 32, 4, 5)
        attn = layer(query, content, content, mask)
        self.assertTrue(shape_equal([3, 5, 10, 32], attn.shape))

    def test_ensemble_independence(self):
        """ Tests whether the class processes ensemble channels independently"""
        query = torch.randn([3,5,10, 64])
        content = torch.randn([3, 5, 30, 16])

        mask = torch.randn([10, 30]) > 0.5
        layer = Attention.MultiHeadedAttention(64, 16, 32, 4, 5)
        attn = layer(query, content, content, mask)

        query_update = torch.randn([3, 1, 10, 64])
        content_update = torch.randn([3, 1, 30, 16])

        query_update = torch.concat([query[:, :-1], query_update], dim=-3)
        content_update= torch.concat([content[:, :-1], content_update], dim=-3)
        update_attn = layer(query_update, content_update, content_update, mask)

        test_a_vals = attn[:, :-1]
        test_b_vals = update_attn[:, :-1]
        self.assertTrue(torch.all(test_a_vals == test_b_vals))



    def test_torchscript_compile(self):
        query = torch.randn([3,5,10, 64])
        content = torch.randn([3, 5, 30, 16])
        mask = torch.randn([10, 30]) > 0.5
        layer = Attention.MultiHeadedAttention(64, 16, 32, 4, 5)
        layer = torch.jit.script(layer)
        attn = layer(query, content, content, mask)

class test_PIMU(unittest.TestCase):
    """
    Test case for the PIMU class
    """
    def test_basic(self):
        """ Test whether a basic PIMU instance works."""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.PIMU(32, 10, 4)
        shape = torch.tensor([3, 5, 10, 32])
        injected = layer(query)
        injected_shape = torch.tensor(injected.shape)
        self.assertTrue(torch.all(shape == injected_shape))
    def test_ensemble(self):
        """ Test whether an ensembled PIMU instance works."""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.PIMU(32, 10, 4, 5)
        injected = layer(query)
        self.assertTrue(shape_equal([3, 5, 10, 32], injected.shape))
    def test_torchscript_compile(self):
        """ Tests whether torchscript compiles on the layer"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.PIMU(32, 10, 4, 5)
        layer = torch.jit.script(layer)
        injected = layer(query)


class test_PISU(unittest.TestCase):
    def test_basic(self):
        """ Tests whether a basic, no frills PISU instance will work"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.PISU(32, 16, 4, 4)
        summary = layer(query)

        summary_shape = torch.tensor(summary.shape)
        expected_shape = torch.tensor([3, 5, 4, 16])
        self.assertTrue(torch.all(summary_shape == expected_shape))
    def test_ensemble(self):
        """ Test whether the ensembled PISU instance works"""
        query = torch.randn([3,5, 10, 32])
        layer = Attention.PISU(32, 16, 20, 4, 5)
        summary = layer(query)
        self.assertTrue(shape_equal([3, 5, 20, 16], summary.shape))
    def test_torchscript_compile(self):
        """ Tests whether torchscript safely compiles"""
        query = torch.randn([3,5, 10, 32])
        layer = Attention.PISU(32, 16, 20, 4, 5)
        layer = torch.jit.script(layer)
        summary = layer(query)

class test_EESA(unittest.TestCase):
    """

    Unit test for the Ensemble Exchange Self Attention
    layer.

    """
    def test_basic(self):
        """ test whether any random tensor makes it through"""
        ensembles, items, heads, d_model = (5, 10, 8, 32)

        query = torch.randn([3, ensembles, items, d_model])
        layer = Attention.EESA(d_model, heads, ensembles)
        output = layer(query)
        self.assertTrue(shape_equal(output.shape, [3, ensembles, items, d_model]))
    def test_masking(self):
        """ test whether the masking function is working properly,
         such that the mask is preventing peering ahead. """
        ensembles, items, heads, d_model = (5, 10, 8, 32)

        query_primary = torch.randn([3, ensembles, items, d_model])
        update = torch.zeros([3,1, items, d_model])
        query_update = torch.concat([query_primary[:, :-1], update], dim=-3)

        layer = Attention.EESA(d_model, heads, ensembles)
        output_a = layer(query_primary)
        output_b = layer(query_update)

        difference = output_a - output_b
        test_bool = torch.all(difference[:, :-1] < 0.001)
        self.assertTrue(test_bool)
    def test_torchscript(self):
        """ Test that torchscript can compile the layer"""
        ensembles, items, heads, d_model = (5, 10, 8, 32)

        query = torch.randn([3, ensembles, items, d_model])
        layer = Attention.EESA(d_model, heads, ensembles)
        layer = torch.jit.script(layer)
        output = layer(query)
        self.assertTrue(shape_equal(output.shape, [3, ensembles, items, d_model]))


class test_LCSA(unittest.TestCase):
    """

    Unit test for the Local Context Self Attention layer.

    """
    def test_basic(self):
        """ test whether it runs at all."""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.LCSA(32, 4, [1, 2, 5, 10])
        local_conditioning = layer(query)
        self.assertTrue(shape_equal([3, 5, 10, 32], local_conditioning.shape))
    def test_ensemble(self):
        """ test whether operating in ensemble mode causes any bugs"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.LCSA(32, 4, [1, 2, 5, 10], ensemble=5)
        local_conditioning = layer(query)
        self.assertTrue(shape_equal([3, 5, 10, 32], local_conditioning.shape))
    def test_torchscript_compiles(self):
        """ test whether torchscript is willing to compile an initialized layer"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.LCSA(32, 4, [1, 2, 5, 10], ensemble=5)
        layer = torch.jit.script(layer)
        local_conditioning = layer(query)
        self.assertTrue(shape_equal([3, 5, 10, 32], local_conditioning.shape))


class test_GLSA(unittest.TestCase):
    """

    The test fixture for GSLA.

    """
    def test_basic(self):
        """ Test whether GLSA works at all"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.GLSA(32, 64, 10, 8)
        glsa = layer(query)
        self.assertTrue(shape_equal(glsa.shape, [3, 5, 10, 32]))
    def test_ensemble(self):
        """ Test whether the ensemble system works properly"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.GLSA(32, 64, 10, 8, 5)
        glsa = layer(query)
        self.assertTrue(shape_equal(glsa.shape, [3, 5, 10, 32]))
    def test_torchscript(self):
        """ Test whether torchscript compiles properly"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.GLSA(32, 64, 10, 8, 5)
        layer = torch.jit.script(layer)
        glsa = layer(query)
        self.assertTrue(shape_equal(glsa.shape, [3, 5, 10, 32]))

class test_GSPU(unittest.TestCase):
    """
    Test fixture for the Global Strategic Processing Unit.

    """
    def test_basic(self):
        """ Test if it runs without any bells or whistles"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.GSPU(32, 32, 10, 8, 8)
        output = layer(query)
        self.assertTrue(shape_equal(output.shape, [3, 5, 10, 32]))
    def test_ensemble(self):
        """ Test if it works when using an ensemble"""
        query  = torch.randn([3, 5, 10, 32])
        layer = Attention.GSPU(32, 16, 10, 8, 4, ensembles=5)
        output = layer(query)
        self.assertTrue(shape_equal(output.shape, [3, 5, 10, 32]))
    def test_with_sublayers(self):
        """ Test if it works when provided a few sublayers """

        sublayers = [Attention.FeedForward(16, 128, 5), Attention.FeedForward(16, 128, 5)]
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.GSPU(32, 16, 10, 8, 4, layers =sublayers, ensembles=5)
        output = layer(query)
        self.assertTrue(shape_equal(output.shape, [3, 5, 10, 32]))
    def test_torchscript(self):
        """ Test if it will compile to torchscript"""

        sublayers = [Attention.FeedForward(16, 128, 5), Attention.FeedForward(16, 128, 5)]
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.GSPU(32, 16, 10, 8, 4, layers =sublayers, ensembles=5)
        layer = torch.jit.script(layer)
        output = layer(query)

        self.assertTrue(shape_equal(output.shape, [3, 5, 10, 32]))

if __name__ == '__main__':
    unittest.main()
