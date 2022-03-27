
# perform imports

import numpy as np
import torch
from torch import nn

import math
import numbers

# perform library imports
from Utility.Torch import Glimpses
from Utility.Torch import Activation

### Head accommodation on the linear layer ###

class Linear(nn.Module):
    """

    A Linear layer allowing head-dependent linear processing of data from shape
    to shape.

    An instance is made by providing a list of head_shapes,
    an input_shape tuple, an output_shape tuple.

    This is then used to initialize a head dependent linear remap
    from input shape to output shape. That will then be accessed
    through the instance call

    It is expected that the input format will be in the form of

    [..., heads, input_shape]

    Returning something of format

    [..., heads, output_shape]


    Letting the head_shape parameter be none will disable it, resulting in broadcasting. Input
    shape, output shape, and head_shapes may all be just an integer, in which case it is
    assumed only a single dimension is involved.

    """

    def __init__(self, input_shape, output_shape, head_shapes=None):
        """

        :param input_shape: The shape of the input. May be an int, or a list/tuple of ints
        :param output_shape: The shape of the output. May be an int, or a list/tuple of ints.
        :param head_shapes: The head dimensions, which come immediately prior to the
            input dimensions. May be an int, or a list/tuple of ints.
        """
        # Super call

        super().__init__()

        # Implicit conversion
        if head_shapes is None:
            head_shapes = []
        elif isinstance(head_shapes, int):
            head_shapes = [head_shapes]
        if isinstance(input_shape, int):
            input_shape = [input_shape]
        if isinstance(output_shape, int):
            output_shape = [output_shape]

        # Create preprocesser and postprocessor. These flatten, and unflatten, the
        # dimensions we care about

        self._preprocessor = lambda x: Glimpses.view(x, input_shape, np.prod(input_shape))
        self._postprocesser = lambda x: Glimpses.view(x, np.prod(output_shape), output_shape)

        # Create kernel and bias. These include head dimensions if provided.

        if head_shapes is not None:
            kernel_shape = [*head_shapes, np.prod(output_shape), np.prod(input_shape)]
            bias_shape = [*head_shapes, np.prod(output_shape)]
        else:
            kernel_shape = [np.prod(output_shape), np.prod(input_shape)]
            bias_shape = [np.prod(output_shape)]

        kernel = torch.zeros(kernel_shape, requires_grad=True)
        kernel = torch.nn.init.kaiming_uniform_(kernel, a=math.sqrt(5))

        bias = torch.zeros(bias_shape, requires_grad=True)
        bias = torch.nn.init.zeros_(bias)

        # Store

        self._kernel = nn.Parameter(kernel)
        self._bias = nn.Parameter(bias)

    def forward(self, tensor):

        # Flatten the relavent dimensions

        tensor = self._preprocessor(tensor)

        # Perform primary processing. Add an extra dimension on the end
        # of the input tensor to handle the matrix multiply, perform
        # matrix multiply then add bias

        tensor = tensor.unsqueeze(-1)
        tensor = self._kernel.matmul(tensor)
        tensor = tensor.squeeze(-1)
        tensor = tensor + self._bias

        # Restore the dimensions
        tensor = self._postprocesser(tensor)

        # Return
        return tensor

### Archival and other data retrieval






class Archivist(nn.Module):

    """
    Description:

    The purpose of this class is to act as a container for holding
    the duel_usage indexer and retrieval specialist classes, which
    work together on the encoder and decoder portion of a model respectively.

    --- attributes ---

    userspace:

    class Indexer: A class, performs the task of indexing an encoder to produce
        an archive
    class Archive: A class. The output of an indexer. Consists of an index, and
        a few functions for retrieval and storage.
    class Access: A class. Accepts a list of archives and allows retrieval
        and even storage of them.

    --- methods ---

    __init__ : builds an Indexer and an Archive
    """
    class Archive(nn.Module):
        """

        Description:

        The purpose of this class is to act as an interface between
        some sort of indexer and a access class. The class is intended
        to hold a record and some sort of index, and should also contain
        the logic to access the index upon forward call.

        The forward call is performed with a sequence of entries known as
        queries, and will return the entries which scored highly in the index.
        The return is designed to be usable only by entities, such as transformers,
        for which the order information is presented does not matter.

        Items scoring above "threshold" are presented as distinct tensor entities.
        Meanwhile, items scoring below "threshold" are lumped together into a final
        entity. This is okay - vl_relu activation is expected to mostly shut off
        irrelevant entities, and relevant entities will increase in score until
        peeling off and becoming active.

        It is HIGHLY recommended that the output of the archive be run through
        a LayerNorm before usage.

        If one was to summarize the logic, one might say something like "This checks what
        words.

        The archive class is entirely stateless, with no parameters involved.

        --- attributes ----

        archive_length: the length of the record underlying the archive.
        record_dim: the dimensionality of each record.
        index_dim: the dimensionality of each index. Queries must match this width.


        _record: The underlying stored record
        _index: The indexes for the record.

        --- methods ---


        """
        @property
        def archive_length(self):
            """ Returns the archive width length"""
            return self._record.shape[-2]
        @property
        def record_dim(self):
            """ Returns the record dimension"""
            return self._record.shape[-1]
        @property
        def index_dim(self):
            """ Returns the index dimension"""
            return self._index.shape[-1]

        def __init__(self, record, index, threshold=0.1):
            """

            Init for Archive.

            Record should be a possibly batched encoding from
            a transformer, given as (..., item, d_model).

            Index should be an index generated from the record. It has
            shape (..., item, d_index) and is of type float16. One index
            per item.

            Threshold is a constant. Anything scoring below this will be
            compressed and mostly discarded.

            :param record: the record
            :param index: The index
            :param threshold: The threshold
            """
            # Start torch

            super().__init__()

            # Store parameters
            self._record = record
            self._index = index
            self._threshold = threshold

        def forward(self, query, train_index=False):
            """

            This will accept a query and return a collection of results
            which were found to immediately correspond to the particular query.

            It might best be thought of as asking "Given this word/encoding, what
            things were immediately relevant to it."

            Because the relevant items might vary depending on the query, it is the
            case that the width of the last dimension is variable. Inactive encodings
            are masked with zero.

            Finally, the last entry, when train_index is active, is designed to be a mostly
            off gradient passthrough layer, which will encourage useful entities to
            turn back on if needed. This being said, it may be faster with this feature
            off.

            :param query: A (..., query, index_dim) tensor representing things which we wish to look at
            :param train_index: A bool. Represents whether or not a gradient recovery feature is engaged.
                This feature may significantly increase computational cost, especially on large text
                corpusus, but may increase accuracy if trained for a small percent of the time.
            :param d_kernel: A model specific parameter, controlling the exponent on the kernel
            """
            # Perform a query lookup. This is executed under the assumption
            # that when you multiply a query by the index, a higher score means
            # a more relevant result.
            #
            # The results with scores below threshold are sorted out of the return,
            # then either averaged together and appended or excluded from the return

            # Make sure type is compatible with index
            query = query.type(torch.float16)


            # Perform scoring.

            score = torch.matmul(query, self._index.transpose(-1, -2).type(torch.float16)).type \
                (torch.float32)  # (..., query, item)

            assert_a = torch.tensor([query.shape[-2], self.archive_length])
            assert_b = torch.tensor(score.shape[-2:])
            assert torch.equal(assert_a, assert_b), (assert_a, assert_b, query.shape)
            passed = score > self._threshold #(..., query, item)


            ###Figure out what the passing records are. These records will be  in a block
            # with shape (..., query, max_items), where max_items is the maximum number
            # of passing items across all queries and nonpassing records have been masked to
            # be zero. The activation of each record by the score should be noted.

            max_passed = passed.sum(dim=-1).max() #(..., query)
            indices = torch.argsort(passed, dim=-1, descending=True)  # I just need to know what indices passed.
            indices = indices[..., :max_passed]  # Anything beyond this failed. Discard

            shape = [-1]*len(indices.shape)
            indices = indices.unsqueeze(-1)
            indices = indices.expand(*shape, self.record_dim) #index includes record access dim

            # index now: (..., query, max_items, d_model)

            records = self._record.unsqueeze(-3) #(..., 1 (query), items, record_dim)
            broadcast_score = score.unsqueeze(-1) #(..., query, items, 1)

            activated_records = records.masked_fill(broadcast_score, 0)  # No failed gradients
            activated_records = torch.nn.functional.relu(score.unsqueeze(-1) ) * activated_records

            passing_records = torch.gather(activated_records, dim=-2, index =indices)  # Keep all indices needed for passing.

            #(..., query, max_items, d_model)

            if train_index:
                ### The failing entities are also incorporated into the return under a single value.

                failed = torch.logical_not(passed)
                failed_number = failed.sum(dim=-1) #(..., query)

                record = self._record.unsqueeze(-3)
                broadcast_failed = failed.unsqueeze(-1)

                failed_records = record.masked_fill(broadcast_failed, 0)
                failed_records = (Activation.vl_relu(score.unsqueeze(-1))) * failed_records  # No dead gradients.
                failed_sum = failed_records.sum(dim=-2) # (..., query, d_model)


                failing_summary = failed_sum/failed_number.unsqueeze(-1)  # Average. (..., query, d_model)
                failing_summary = failing_summary.unsqueeze(-2) #Item dim. (..., query, 1 (item), d_model)

                output = torch.concat([passing_records, failing_summary], dim=-2)
            else:
                output = passing_records

            # Perform layernorm. Return.
            #output = self._layernorm(output)
            return output


    class Indexer(nn.Module):
        """
        Description:

        The purpose of this class is to store information which has passed
        through the encode part of an encoder-decoder into a format which
        enables easy and efficient lookup of information relative to a particular
        query sequence.

        """
        def __init__(self, d_model, d_index, threshold=0.1):

            # Start torch
            super().__init__()

            #Store standing

            self._threshold = threshold

            # Create index generating code
            self._ff1 = Linear(d_model, 3* d_model)
            self._ff2 = Linear(3 * d_model, d_index)

        def forward(self, record):
            """
            Creates an Archive from an encoding Record. This consists
            of running the record through a sequence of layers to
            generate an index of the appropriate shape.

            :param record: The output of a transformer-style encoder.
                Should consists of (..., items, d_model)
            :return:
                An archive class. This consists of a class named archive
                possessing a forward method capable of returning results
                which look relevant.
            """

            # Generate index
            index = record
            index = self._ff1(index)
            index = Activation.vl_relu(index)
            index = self._ff2(index)

            # Make Archive

            return Archivist.Archive(record, index, self._threshold)
    class Retrieval(nn.Module):
        """
        Description:

        The purpose of this class is to allow the accessing
        of collections of archives.




        """


class Transformer(nn.Module):
    __permitted = (None, "lower", "upper")

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        assert value in self.__permitted, "mask cannot be set to this"
        self._mask = value

    def __init__(self, channel_dim, head_width, mask=None):

        """

        Accepted mask is "lower", "upper", or none

        """

        # Spin up torch
        super().__init__()

        # Create action generators
        QueryGen = LinearReshape(channel_dim, (head_width, channel_dim))
        KeyGen = LinearReshape(channel_dim, (head_width, channel_dim))
        ValGen = LinearReshape(channel_dim, (head_width, channel_dim))

        CollapseGen = LinearReshape((head_width, channel_dim), channel_dim)

        # Create actions. Note the swap is needed to get the head in front of the items.

        self._query = lambda x: QueryGen(x).swapdims(-2, -3)
        self._key = lambda x: KeyGen(x).swapdims(-2, -3)
        self._value = lambda x: ValGen(x).swapdims(-2, -3)
        self._dehead = lambda x: CollapseGen(x.transpose(-2, -3))

        self.mask = mask

    def forward(self, query, content, mask=None):
        # Create query, key, value

        query = self._query(query)
        key = self._key(content).swapdims(-1, -2)
        value = self._value(content)

        # Create focus matrix. Mask. Softmax.

        focus = query.matmul(key)
        focus_dims = focus.shape[-2:]
        if mask is None:
            # Runs only if not provided a mask.
            if self.mask == "lower":
                mask = torch.tril(torch.ones(focus_dims))
                focus = focus.masked_fill(mask == 0, -1e9)
            if self.mask == "upper":
                mask = torch.triu(torch.ones(focus_dims))
                focus = focus.masked_fill(mask == 0, -1e9)

        focus = F.softmax(focus, dim=-1)

        # Apply focus matrix to values. Then compact head

        output = focus.matmul(value)
        output = self._dehead(output)

        return

