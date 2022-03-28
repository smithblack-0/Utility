# perform imports

import numpy as np
import torch
from torch import nn

import math
import numbers
import uuid
import warnings

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

        # Flatten the relevent dimensions

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
        some sort of indexer and an access class. The class is intended
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
        embeddings are immediately relevant to a particular query from an entire text
        corpus"

        The archive class is entirely stateless, with no parameters involved.

        --- attributes ----

        archive_length: the length of the record underlying the archive.
        record_dim: the dimensionality of each record.
        index_dim: the dimensionality of each index. Queries must match this width.


        _record: The underlying stored record
        _index: The indexes for the record.
        _
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
        @property
        def id(self):
            """ Returns the id"""
            return self._id

        def __init__(self, record, index, id, threshold=0.1):
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
            :param id: A id of some sort, presumably a uuid. Uniquely identifies
                what indexer was responsible for making this archive
            """
            # Start torch

            super().__init__()
            # Perform basic sanity checks

            assert torch.is_tensor(record), "record was not tensor"
            assert torch.is_tensor(index), "index was not tensor"
            assert torch.equal(torch.tensor(record.shape[:-2]), torch.tensor(index.shape[:-2])), \
                ("Index, record shapes incompatible", index.shape, record.shape)

            # Store parameters
            self._record = record
            self._index = index
            self._threshold = threshold
            self._id = id

        def forward(self, query : torch.tensor, train_index: bool=False):
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

            :param query: A (..., head,  query, index_dim) tensor representing things which we wish to look at.
                The items on query, head represent what we actually are investigating, and will all be collapsed
                into one final answer.
            :param train_index: A bool. Represents whether or not a gradient recovery feature is engaged.
                This feature may significantly increase computational cost, especially on large text
                corpusus, but may increase accuracy if trained for a small percent of the time.
            :return : A tensor of shape (..., query, max_item, d_model). This represents for each head collection
                in (head, query), what items were found to be interesting to the query, all of which are returned
                in sequence.
            """
            ### Perform a query lookup. This is executed under the assumption
            # that when you multiply a query by the index, a higher score means
            # a more relevant result.
            #
            # The results with scores below threshold are sorted out of the return,
            # then either averaged together and appended or excluded from the return

            # Perform basic sanity testing and conversion.

            assert self.index_dim == query.shape[-1], "Length of query index dimension does not match archive dimension"
            if len(query.shape) == 1:
                query = query.unsqueeze(0)
            if len(query.shape ) == 2:
                query = query.unsqueeze(0)



            ### Perform scoring. Then reshape to collapse heads. Do this by taking
            # the query, transforming it to the right shape for broadcast, doing the same
            # thing for score, then matrix multiplying by the index dimension. Following this,
            # route all the heads onto the item dimension.

            query = query.type(torch.float16)  # (..., head, query, d_index)
            index = self._index.transpose(-1, -2).type(torch.float16)  # (..., d_index, item)
            index = index.unsqueeze(-3)  # (...,  1 (head), d_index, item)
            score = torch.matmul(query, index).type(torch.float32)  # (..., head, query, item)
            score = score.transpose(-3, -2).flatten(-2) #(..., query, head*item) = (query, new_item)

            # Grade the score. Create bools.
            passed = score > self._threshold  # (...  query, new_item)

            ### Sort out what the passing records are. Collect these into a single tensor
            # These records will be in a block with shape (..., query, max_items)
            # where max_items is the maximum number of passing items across all
            # queries and nonpassing records have been masked to be zero.
            # The activation of each record by the relu of the score should also be noted:
            # this ensures smooth on/off behavior.

            max_passed = passed.sum(dim=-1).max()  # now a 0d tensor.
            indices = torch.argsort(passed, dim=-1, descending=True)  # I just need to know what indices passed.
            indices = indices[..., :max_passed]  # (..., query, max_item), Anything beyond this failed. Discard

            #We will be using the indices for a gather. We need to account for the additional
            #record d_model dimension. Do this with a memory-efficient expand.
            shape = [-1] * len(indices.shape)
            indices = indices.unsqueeze(-1)
            indices = indices.expand(*shape, self.record_dim) #(..., query, max_item, d_model)

            ## Fetch and create the records we will use. Due to the existance of heads,
            # we tile the record along the item axis a sufficient amount to account for
            # all heads.

            records = self._record.unsqueeze(-3)  # (...,  1 (query), items, record_dim)
            shape = [1] * len(records.shape)
            shape[-2] = query.shape[-3]  # Extend the records to account for multiple heads.
            records = records.tile(shape)

            ##Now, go ahead and mask the records. The masked records should be multiplied by the
            #relu of their scores. This will ensure that shutting down the score will
            #reduce the effect of the record, giving the index something to train off of.



            broadcast_score = score.unsqueeze(-1)  # (..., query, items, 1 (d_model))
            activated_records = records.masked_fill(broadcast_score, 0)  # No failed gradients
            activated_records = torch.nn.functional.relu(score.unsqueeze(-1)) * activated_records

            #Perform the actual gather.
            passing_records = torch.gather(activated_records, dim=-2,
                                           index=indices)  # Keep all indices needed for passing.

            # (..., query, max_items, d_model)

            if train_index:
                ### The failing entities are also incorporated into the return under a single value.

                failed = torch.logical_not(passed)
                failed_number = failed.sum(dim=-1)  # (..., query)
                broadcast_failed = failed.unsqueeze(-1)

                failed_records = records.masked_fill(broadcast_failed, 0)
                failed_records = (Activation.vl_relu(score.unsqueeze(-1))) * failed_records  # No dead gradients.
                failed_sum = failed_records.sum(dim=-2)  # (..., query, d_model)

                failing_summary = failed_sum / failed_number.unsqueeze(-1)  # Average. (..., query, d_model)
                failing_summary = failing_summary.unsqueeze(-2)  # Item dim. (..., query, 1 (item), d_model)

                output = torch.concat([passing_records, failing_summary], dim=-2)
            else:
                output = passing_records

            # Return result.
            return output

    class Indexer(nn.Module):
        """
        Description:

        The purpose of this class is to store information which has passed
        through the encode part of an encoder-decoder into a format which
        enables easy and efficient lookup of information relative to a particular
        query sequence.

        It goes out of it's way to ensure each index has both local information,
        from the immediate constructor, along with global information gathered from
        Dropout is executed post-index projection, while dimensions remain high.

        It MUST be the case that all archives with the same id have the same d_model
        and d_index
        """

        def __init__(self, d_model, d_index, projection_multiplier= 3, id=None,
                     metahead_width = 20, threshold=0.1, dropout = 0.1):
            # Start torch
            super().__init__()

            # Store standing values

            self._threshold = threshold

            # Create index generation layers
            self._ff1 = Linear(d_model, projection_multiplier * d_model)
            self._ff2 = Linear(projection_multiplier * d_model, d_index)

            #Create the metainfo generation layers

            self._metainfo1 = Linear(d_index, (metahead_width, projection_multiplier*d_index)) # Generation of meta reduction heads
            self._metainfo2 = Linear(projection_multiplier*d_index, d_index, metahead_width) # Post ReLu reduction.
            self._metainfo3 = Linear(metahead_width, 1) #Final reduction layer

            #Create the layernorm finale layer, and dropout layer
            self._LayerNorm = nn.LayerNorm(d_index)
            self._dropout = nn.Dropout(dropout)

            #Assign myself an id
            if id is None:
                self._id = uuid.uuid1()
            else:
                self._id = id

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

            # Generate index by using the feedforward layers. This is done
            # with a series of feedforwards much like a transformer would
            # experience.
            index = record
            index = self._ff1(index)
            index = self._dropout(index)
            index = Activation.vl_relu(index)
            index = self._ff2(index)


            # Salt index with metainfo marker. This is created by
            # projecting the index to produce extra heads, transforming those
            # heads under a ReLU, taking the maximum across the index, and finally
            # combining all the heads to produce a final metavector.
            #
            # This is then broadcast over the index, giving all of them info on things the model
            # considered important.

            metatensor = self._metainfo1(index)
            metatensor = Activation.vl_relu(metatensor)
            metatensor = self._metainfo2(metatensor)# (..., items, metahead, index)
            metatensor = metatensor.max(dim=-3).transpose(-1,-2) #(..., index, metahead)
            metatensor = self._metainfo3(metatensor).transpose(-1, -2) #(..., 1, index_dim)

            index = index + metatensor
            index = self._LayerNorm(index)
            index = index.type(torch.float16)
            # Make Archive

            return Archivist.Archive(record, index, self._id, self._threshold)

    class Retrieval(nn.Module):
        """
        Description:

        The purpose of this class is to allow the accessing
        of information in the decoder portion of a model from
        a collection of provided archives in an efficient
        manner.

        In particular, it is designed to contain
        the fetch logic required to generate heads and
        perform any other needed tasks internally. The
        output contains the source-marked varieties
        of
        """
        def __init__(self, d_queries, d_model, retrieval_heads = 5, dropout = 0.1, suppress_errors=False):
            """
            The setup function is rather sparse. It is the case
            that the initialization of actual head handlers is

            :param d_queries: The expected d_model dimension of the incoming queries
            :param retrieval_heads: How many heads each item will possses
            """
            #Start torch

            super().__init__()
            #Store the constants which tell what the input and output should look like

            self._dquery = d_queries
            self._dmodel = d_model

            #Store the constants which say something about how the information will be looked up
            self._heads = retrieval_heads
            self._dropout = nn.Dropout(dropout)

            #Setup the archive interface cache

            self._archive_interface = {}

            #Store warning constants.

            self._suppress_errors = suppress_errors
            self._first_run = True

        def generate_interface(self,  d_index, d_model):
            """
            A function to develop the interface transforming from and to my encoding
            space to a particular archive space.

            Internal

            """

            archive_encoder = Linear(self._dquery, (self._heads, d_index))
            archive_encoder = lambda x : archive_encoder(x).transpose(-2, -3) #(..., head, query, d_index)
            archive_decoder = Linear(d_model, self._dmodel) #Back to my dimensions
            return archive_encoder, archive_decoder
        def forward(self, query : torch.tensor, archives : list):
            """


            :param query: A vector in (..., query, d_queries) format, asking about information related
                to each query from the archives
            :param archives: A list of archives, defined by indexers. May change in quantity or
                type at any point, but do note that archives generated by different indexers
                will have to learn their interfaces from scratch.
            :return: A tensor of shape (..., query, dependent, d_model), where dimension dependent
                depends on exactly how successful and helpful the model lookup was.
            """
            output_tensor = []
            for archive in archives:
                #Handle the generation of new interfaces for novel archives. Also
                #warn if relavent, in case someone is generating a new interface each time
                #or forgot to load their id's

                if archive.id not in self._archive_interface:
                    if not self._first_run and not self._suppress_errors:
                        warnings.warn("A Retriever is generating an interface on other than the first call")
                    self._archive_interface[archive.id] = self.generate_interface(archive.index_dim, archive.record_dim)
                #Fetch the archive encoder and decoder.
                #For each archive, encode the query, pass it in, then decode the output

                archive_encoder, archive_decoder = self._archive_interface[archive.id]
                encoded_query = archive_encoder(query)
                archive_return = archive(encoded_query)
                decoded_archive = archive_decoder(archive_return)
                #Store the decoded result. This will be concatenated
                output_tensor.append(decoded_archive)
            #Finish by concatenating along the additional max_item dimension
            output = torch.concat(output_tensor, dim=-2) #(..., query, results, d_model)
            return output



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
