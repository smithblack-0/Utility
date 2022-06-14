import unittest
import torch

import Utility.Torch.Archive.Archivist as Archivist


class testArchive(unittest.TestCase):
    """

    This is the test case for the
    archive class of Archivist

    """

    def test_Constructor(self):
        """ Tests the constructor and return parameters in a few different configurations"""

        record = torch.randn([30, 10, 20])
        index = torch.randn([30, 10, 4])
        Archivist.Archive(record, index)

    def test_Basic_Truth_Lookup(self):
        """ Tests whether a lookup may be performed, and if the shape is correct."""
        # create test fixtures
        record = torch.randn([30, 10, 20])
        index = torch.ones([30, 10, 4])
        query = torch.ones([30, 1, 34, 4])
        test_fixture = Archivist.Archive(record, index)

        expected_normal_shape = torch.tensor([30, 34, 10, 20])
        expected_training_shape = torch.tensor([30, 34, 11, 20])

        # perform tests
        normal_test, _ = test_fixture(query)
        training_test, _ = test_fixture(query, True)

        normal_result = torch.equal(torch.tensor(normal_test.shape), expected_normal_shape)
        training_result = torch.equal(torch.tensor(training_test.shape), expected_training_shape)

        # Run assert
        self.assertTrue(normal_result, "Nonpropogating lookup came out with wrong shape")
        self.assertTrue(training_result, "Propogating lookup came out with wrong shape")

    def test_Basic_False_Lookup(self, ):
        """ Tests behavior when everything is found to be false. Tests shutoff works"""
        # create test fixtures
        record = torch.randn([30, 10, 20])
        index = torch.ones([30, 10, 4])
        query = -torch.ones([30, 1, 34, 4])
        test_fixture = Archivist.Archive(record, index)

        expected_normal_shape = torch.tensor([30, 34, 0, 20])
        expected_training_shape = torch.tensor([30, 34, 1, 20])

        # perform tests
        normal_test, _ = test_fixture(query)
        training_test, _ = test_fixture(query, True)

        normal_result = torch.equal(torch.tensor(normal_test.shape), expected_normal_shape)
        training_result = torch.equal(torch.tensor(training_test.shape), expected_training_shape)

        # Run assert
        self.assertTrue(normal_result, "Nonpropogating lookup came out with wrong shape")
        self.assertTrue(training_result, "Propogating lookup came out with wrong shape")

    def test_Head_Lookup(self):
        """ Tests whether a lookup involving using the heads may be performed, and if the shape is correct."""
        # create test fixtures
        record = torch.randn([28, 10, 20])
        index = torch.ones([28, 10, 4])
        query = torch.ones([28, 2, 34, 4])
        test_fixture = Archivist.Archive(record, index)

        expected_normal_shape = torch.tensor([28, 34, 20, 20])
        expected_training_shape = torch.tensor([28, 34, 21, 20])

        # perform tests
        normal_test, _ = test_fixture(query)
        training_test, _ = test_fixture(query, True)

        normal_result = torch.equal(torch.tensor(normal_test.shape), expected_normal_shape)
        training_result = torch.equal(torch.tensor(training_test.shape), expected_training_shape)

        # Run assert
        self.assertTrue(normal_result, "Nonpropogating lookup came out with wrong shape")
        self.assertTrue(training_result, "Propogating lookup came out with wrong shape")

    def test_logic(self):
        """ Test whether the actual underlying logic words. """
        # create test fixtures
        record: torch.Tensor = torch.arange(12)
        record = record.view(12, 1)

        index: torch.Tensor = torch.concat([torch.ones([6, 1]), -torch.ones([6, 1])], dim=0)

        query = torch.ones([2])
        query[1::2] = -query[1::2]
        query = query.view(2, 1)

        expected = torch.stack([torch.arange(6), torch.arange(6, 12)], dim=0).type(torch.float32)

        test_feature = Archivist.Archive(record, index)
        # Run tests

        test_output, _ = test_feature(query)
        test_bool = torch.equal(test_output.squeeze(), expected)
        self.assertTrue(test_bool, "Logic failed")


class test_Indexer(unittest.TestCase):
    """
    This is the test fixture for the indexer portion of Archivist
    """

    def test_Constructor(self):
        """ tests whether the constructor will run or fail successfully in a variety of situations"""
        Archivist.Indexer(512, 60)
        Archivist.Indexer(512, 60, 5, 312, 0.4, 0.5)

    def test_no_item_dim_Create(self):
        """ tests whether an implicit basic archive can be constructed"""

        # Create test fixtures
        record: torch.Tensor = torch.randn([512])
        indexer: Archivist.Indexer = Archivist.Indexer(512, 60, id=4123)

        # Run test
        test_result: Archivist.Archive = indexer(record)

        # check results

        self.assertTrue(test_result.archive_length == 1, "Archive length was incorrect")
        self.assertTrue(test_result.index_dim == 60, "Index length was incorrect")
        self.assertTrue(test_result.record_dim == 512, "final d_model was incorrect")
        self.assertTrue(test_result.id == 4123, "Id was incorrect")

    def test_Nonbatched_Record_Create(self):
        """ tests whether or not an unbatched record can be succesfully constructed"""

        # create test fixtures
        record: torch.Tensor = torch.randn([32, 512])
        indexer: Archivist.Indexer = Archivist.Indexer(512, 20)

        # Run test
        test_result: Archivist.Archive = indexer(record)

        # Check results

        self.assertTrue(test_result.record_dim == 512, "record length incorrect")
        self.assertTrue(test_result.archive_length == 32, "archive length incorrect")
        self.assertTrue(test_result.index_dim == 20, "index dim incorrect")

    def test_Batching_Archive_Create(self):
        """ tests whether or not an indexer can successfully create an arbitrarily batched archive"""

        #create test fixtures

        record: torch.Tensor = torch.zeros([13,20, 24, 512])
        indexer: Archivist.Indexer = Archivist.Indexer(512, 64)

        #Run tests

        test_result: Archivist.Archive = indexer(record)

        #Check results

        self.assertTrue(test_result.record_dim == 512, "record d_model incorrect")
        self.assertTrue(test_result.index_dim == 64, "index d_model incorrect")
        self.assertTrue(test_result.archive_length == 24, "archive item lengths incorrect")

class testRetrieval(unittest.TestCase):

    pass