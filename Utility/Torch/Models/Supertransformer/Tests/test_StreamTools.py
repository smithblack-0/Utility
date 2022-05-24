"""

The test module for the streamtools section.


"""
from typing import Dict

import torch
import unittest

from Utility.Torch.Models.Supertransformer import StreamTools


class test_stream_tensor(unittest.TestCase):
    """

    A test case for testing the stream tensor. Stream
    tensors are a collection of related tensors which
    may be operated on in parallel.

    """
    def setUp(self) -> None:
        self.test_stream = {"channel1" : torch.randn([10, 20, 4]),
                       "channel2" : torch.randn([30, 20])}
        self.test_losses = {"loss1" : torch.randn(1)}
        self.test_metrics = {"metric1": [torch.randn([20, 3])]}

    def test_build(self):
        """Tests the tensor constructor is working"""
        tensor = StreamTools.StreamTensor(self.test_stream, self.test_losses, self.test_metrics)
    #Element manipulation methods
    def test_properties(self):
        """Tests the properties work properly on the tensor"""
        tensor = StreamTools.StreamTensor(self.test_stream, self.test_losses, self.test_metrics)
        self.assertTrue(tensor.names == self.test_stream.keys())
        self.assertTrue(tensor.stream == self.test_stream)
        self.assertTrue(tensor.losses == self.test_losses)
        self.assertTrue(tensor.metrics == self.test_metrics)
    def test_isolate(self):
        """ Tests the isolate method"""
        tensor = StreamTools.StreamTensor(self.test_stream, self.test_losses, self.test_metrics)
        test_tensor = tensor.clone()
        item = test_tensor.isolate(['channel1'])
        self.assertTrue(torch.all(item[0] == self.test_stream['channel1']))
    #Stream Manipulation Methods
    def test_branch(self):
        """ Tests branching works okay. This makes a new branch"""
        tensor = StreamTools.StreamTensor(self.test_stream, self.test_losses, self.test_metrics)
        branch_tensor = tensor.branch(['channel1'])
        expected = {'channel1' : self.test_stream['channel1']}
        self.assertTrue(branch_tensor.stream == expected)
        self.assertTrue(branch_tensor.losses == {})
        self.assertTrue(branch_tensor.metrics == {})
    def test_discard(self):
        """ Tests discard works okay. This maintains the branch"""
        tensor = StreamTools.StreamTensor(self.test_stream, self.test_losses, self.test_metrics)
        discard_tensor = tensor.discard(['channel1'])
        expected_stream = {'channel2': self.test_stream['channel2']}
        expected_losses = self.test_losses
        expected_metrics = self.test_metrics
        self.assertTrue(discard_tensor.stream == expected_stream)
        self.assertTrue(discard_tensor.losses == expected_losses)
        self.assertTrue(discard_tensor.metrics == expected_metrics)
    def test_keeponly(self):
        """ Test that the keeponly method works correctly. This maintains the branch"""
        tensor = StreamTools.StreamTensor(self.test_stream, self.test_losses, self.test_metrics)
        keep_tensor = tensor.keeponly(['channel1'])
        expected_stream = {'channel1' : self.test_stream['channel1']}
        expected_losses = self.test_losses
        expected_metrics = self.test_metrics
        self.assertTrue(keep_tensor.stream == expected_stream)
        self.assertTrue(keep_tensor.losses == expected_losses)
        self.assertTrue(keep_tensor.metrics == expected_metrics)
    def test_split(self):
        tensor = StreamTools.StreamTensor(self.test_stream, self.test_losses, self.test_metrics)
        split_directive = {'channel2': (-1, [5, 15])}
        tensor = tensor.split(split_directive)

class test_StreamMergerHelper(unittest.TestCase):
    """
    StreamMerger uses a particular subclass to help it
    merge streams. This tests that that class is correctly
    functioning.
    """

    def test_constructor(self):
        """ test if the constructor works properly"""
        test_items  = {}
        test_items['item1'] = [torch.randn([10, 20]), torch.randn([10, 20])]
        test_items['item2'] = [torch.randn([3, 5]), torch.randn(3, 5)]
        test_items['item3'] = [torch.randn([4, 1]), torch.randn([4, 1])]
        test = StreamTools._MergeHelper(test_items)
    def test_reductive_broadcast(self):
        """ Test if the broadcast mechanism works properly for reduction"""
        broadcast_case1 = [torch.randn([10, 20, 5]), torch.randn([5])]
        broadcast_case2 = [torch.randn([10]), torch.randn([3, 5, 10])]

        item1, item2 = broadcast_case1
        expected_broadcast_results1 = torch.stack([item1, item2.expand(10, 20, -1)])

        item1, item2 = broadcast_case2
        expected_broadcast_results2 = torch.stack([item1.expand(3, 5, -1), item2])

        test_instance = StreamTools._MergeHelper({})
        results1 = torch.stack(test_instance.reduce_broadcast(broadcast_case1))
        results2 = torch.stack(test_instance.reduce_broadcast(broadcast_case2))

        test_bool1 = torch.all(results1 == expected_broadcast_results1)
        test_bool2 = torch.all(results2 == expected_broadcast_results2)
        self.assertTrue(test_bool1)
        self.assertTrue(test_bool2)
    def test_concative_broadcast(self):
        """ Tests the broadcast mechanism works properly when concatenating"""
        concat_case1 = [torch.randn([10]), torch.randn([30, 10, 2])]
        concat_case2 = [torch.randn([10, 20]), torch.randn([30, 10, 20])]

        item1, item2 = concat_case1
        concat_result1 = [item1.expand([30, 10, -1]), item2]

        item1, item2 = concat_case2
        concat_result2 = [item1.expand([30, -1, -1]), item2]

        test_instance = StreamTools._MergeHelper({})
        test_result1 = test_instance.concat_broadcast(concat_case1)
        test_result2 = test_instance.concat_broadcast(concat_case2, concat_dim=-2)

        result_bool1 = torch.all(torch.concat(test_result1, dim=-1) == torch.concat(concat_result1, dim=-1))
        result_bool2 = torch.all(torch.concat(test_result2, dim=-2) == torch.concat(concat_result2, dim=-2))

        self.assertTrue(result_bool1)
        self.assertTrue(result_bool2)
    def test_sum_reduce(self):
        """ Tests the sum reduction"""
        test_items  = {}
        test_items['item1'] = [torch.randn([10, 20]), torch.randn([10, 20])]
        test_items['item2'] = [torch.randn([5]), torch.randn(3, 5)]
        test_items['item3'] = [torch.randn([4, 1]), torch.randn([4, 1])]
        test = StreamTools._MergeHelper(test_items)
        test.sum()
        assert test.reduced
    def test_mean_reduce(self):
        """ Tests the mean reduction"""
        test_items  = {}
        test_items['item1'] = [torch.randn([10, 20]), torch.randn([10, 20])]
        test_items['item2'] = [torch.randn([5]), torch.randn(3, 5)]
        test_items['item3'] = [torch.randn([4, 1]), torch.randn([4, 1])]
        test = StreamTools._MergeHelper(test_items)
        test.mean()
        assert test.reduced
    def test_min_reduce(self):
        """ Tests the min reduction"""
        test_items  = {}
        test_items['item1'] = [torch.randn([10, 20]), torch.randn([10, 20])]
        test_items['item2'] = [torch.randn([5]), torch.randn(3, 5)]
        test_items['item3'] = [torch.randn([4, 1]), torch.randn([4, 1])]
        test = StreamTools._MergeHelper(test_items)
        test.min()
        assert test.reduced

    def test_max_reduce(self):
        """ Tests the max reduction"""
        test_items  = {}
        test_items['item1'] = [torch.randn([10, 20]), torch.randn([10, 20])]
        test_items['item2'] = [torch.randn([5]), torch.randn(3, 5)]
        test_items['item3'] = [torch.randn([4, 1]), torch.randn([4, 1])]
        test = StreamTools._MergeHelper(test_items)
        test.max()
    def test_median_reduce(self):
        """ Tests the median reduction"""
        test_items  = {}
        test_items['item1'] = [torch.randn([10, 20]), torch.randn([10, 20])]
        test_items['item2'] = [torch.randn([5]), torch.randn(3, 5)]
        test_items['item3'] = [torch.randn([4, 1]), torch.randn([4, 1])]
        test = StreamTools._MergeHelper(test_items)
        test.median()
        assert test.reduced

    def test_first_reduce(self):
        """ Tests the first reduction"""
        test_items  = {}
        test_items['item1'] = [torch.randn([10, 20]), torch.randn([10, 20])]
        test_items['item2'] = [torch.randn([5]), torch.randn(3, 5)]
        test_items['item3'] = [torch.randn([4, 1]), torch.randn([4, 1])]
        test = StreamTools._MergeHelper(test_items)
        test.first()
        assert test.reduced

    def test_concat_reduce(self):
        """ Tests the concat reduction"""
        test_items  = {}
        test_items['item1'] = [torch.randn([10, 20]), torch.randn([10, 20])]
        test_items['item2'] = [torch.randn([5]), torch.randn(3, 20)]
        test_items['item3'] = [torch.randn([4, 1]), torch.randn([4, 1])]
        test = StreamTools._MergeHelper(test_items)
        test.concat(-1)
        assert test.reduced

class test_StreamMerger(unittest.TestCase):
    """
    StreamMerger is a object which is capable of merging multiple
    streams together, then returning a final stream. This tests it
    """
    def setUp(self) -> None:
        base_stream_items = {}
        base_stream_items['item1'] = torch.randn([10,5, 30])
        base_stream_items['item2'] = torch.randn([30, 4])
        base_stream_items['item3'] = torch.randn([5, 4])

        base_losses = {'loss1' : torch.randn([10])}
        base_metrics = {'metric1' : [torch.randn([10])]}

        self._base_stream = base_stream_items
        self._base_losses = base_losses
        self._base_metrics = base_metrics
        self._basic = StreamTools.StreamTensor(self._base_stream, self._base_losses, self._base_metrics)
    def test_reduction(self):

        reduce_stream_items = {}
        reduce_stream_items['item1'] = torch.randn([10, 5, 30])
        reduce_stream_items['item2'] = torch.randn([4])
        reduce_stream_items['item3'] = torch.randn([4])

        reductive_losses = self._base_losses.copy()
        reductive_losses['loss2'] = torch.randn([30])

        reductive_metrics = self._base_metrics.copy()
        reductive_metrics['metric2'] = torch.randn([30])

        base_stream = self._basic
        reductive_stream = StreamTools.StreamTensor(reduce_stream_items, reductive_losses, reductive_metrics)

        Merger = StreamTools.StreamMerger([base_stream, reductive_stream])
        Merger.stream.sum()
        Merger.losses.mean()
        final = Merger.build()

        for name in reduce_stream_items.keys():
            self.assertTrue(name in final.stream)
        for name in reductive_losses.keys():
            self.assertTrue(name in final.losses)
        for name in reductive_metrics.keys():
            self.assertTrue(name in final.metrics)
    def test_concat(self):

        concat_stream_items = {}
        concat_stream_items['item1'] = torch.randn([10, 5, 30])
        concat_stream_items['item2'] = torch.randn([2])
        concat_stream_items['item3'] = torch.randn([2])

        concat_losses = self._base_losses.copy()
        concat_losses['loss2'] = torch.randn([30])

        concat_metrics = self._base_metrics.copy()
        concat_metrics['metric2'] = torch.randn([30])

        base_stream = self._basic
        reductive_stream = StreamTools.StreamTensor(concat_stream_items, concat_losses, concat_metrics)

        Merger = StreamTools.StreamMerger([base_stream, reductive_stream])
        Merger.stream.concat(-1)
        Merger.losses.mean()
        final = Merger.build()

        for name in concat_stream_items.keys():
            self.assertTrue(name in final.stream)
        for name in concat_losses.keys():
            self.assertTrue(name in final.losses)
        for name in concat_metrics.keys():
            self.assertTrue(name in final.metrics)