import torch
import numpy as np
import torch_sparse
from torch import nn



class ParamMemPool(nn.Module):
    """
    A location in which parameters and associated tensors may end up being stored.

    -- methods --

    create(size, dtype): creates a new parameter pool of the indicated
    register(SparseParameter): Mutually registers a sparse parameter with the pool

    """


    ### Memory management functions ###
    def _mask_2_addresses(self, mask):
        """
        Converts a mask to its index form

        :param mask: A mask whose last dimension we wish to convert
        :return: A list of indices.
        """

        indices = torch.arange(mask.shape[-1])
        indices = indices.masked_select(mask)
        return indices
    def _get_addresses(self, index):
        """
        Lookup the actual addresses in memory corrosponding to a particular index sequence

        Does not check if the index entry exists.
        """

        #Create the index found boolean matrix
        self_index = self.index #(M, 3)
        self_index.unsqueeze(0) #(1, M, 3)
        index_broadcast = index.unsqueeze(1) #(N, 1, 3)
        index_found_bool = (self_index == index_broadcast).all(dim=-1) #(N, M)
        addresses = self._mask_2_addresses(index_found_bool) #(N, 1)
        addresses = addresses.squeeze()

        return addresses

    def _get_partition(self, index):
        """
        Returns a function which will split any incoming
        tensorlike into regions of in, or not in, memory
        """

        # Create the index found boolean matrix
        self_index = self.index  # (M, 3)
        self_index.unsqueeze(0)  # (1, M, 3)
        index_broadcast = index.unsqueeze(1)  # (N, 1, 3)
        index_found_bool = (self_index == index_broadcast).all(dim=-1)  # (N, M)

        # Create the partition function. This splits the input into the region where addresses

        def partition(item):
            index_found_anywhere = index_found_bool.any(dim=-1)
            index_not_found = torch.logical_not(index_found_anywhere)
            return item[index_found_anywhere], item[index_not_found]

        return partition
    def _allocate_index(self, index):
        """
        Allocates a quantity of memory to index

        :param index: the index to open up
        :return: The addresses for the index
        """

        #Get length inactive indices, and find the corrosponding addresses to activate
        length = index.shape[0]
        inactive = self.allocated == False
        addresses = self._mask_2_addresses(inactive)
        if addresses.shape[0] < length:
            raise MemoryError("Amount of remaining parameter memory is less than size of allocation")
        addresses = addresses[:length]

        #Go set these as active, and return
        self.allocated[addresses] = True
        self.index[addresses] = index[:]

        return addresses
    def _release_addresses(self, addresses):
        """
        Simply releases a sequence of held addresses.
        """

        self.allocate[addresses] = False

    def _set_addresses(self, index, value):
        """

        sets the addresses in memory corrosponding to a particular index -value sequence

        if value is none, frees that memory.
        If index entries are present that have not been allocated before, allocate them
        If index entries are present that have been allocated, set them.
        """

        #Get address locations for currently located entries, along with anything not in memory
        with torch.no_grad()
            partition = self._get_partition(index)
            if value is None:
                #Find the entries that exist.
                found_index, missing_index = partition(index)
                found_addresses = self._get_addresses(found_index)
                self._release_addresses(found_addresses)
            else:

                found_index, missing_index = partition(index)
                found_value, missing_value = partition(value)
                found_addresses = self._get_addresses(found_index)

                #Allocate memory for the unallocated addresses
                new_addresses = self._allocate_index(missing_index)

                #Make final dispatch tensor
                final_index = torch.concat([found_index, missing_index], dim=0)
                final_addresses = torch.concat([found_addresses, new_addresses], dim=0)
                final_values = torch.concat([found_value, missing_value], dim=0)

                #Store in memory

                self.index[final_addresses] = final_index
                self.values[final_addresses] = final_values

    ### Sparse rebuild method ###
    def update(self):
        for key in self.sparse.items():
            index = self.get_index(key)
            value = self.get_values(key)
            self.sparse[key] = torch_sparse.SparseTensor(row=index[1], col=index[2], value=value)


    ### External interface methods ###
    def set(self, id, index, value):
        """
        Sets items associated with the given id
        to value.

        :param id: The id to set to
        :param index: The index to set to
        :param value: The value to set to
        """

        #Transform into full coordinate notation
        new_index = torch.full([index.shape[0], 1], id)
        new_index = torch.concat([new_index, index], dim=-1)

        #Set items, then update parameter
        self._set_addresses(new_index, value)

    def get_index(self, id):
        """ Given an id, returns the index entries associated"""

        index_row = self.index[:, 0]
        index_mask = index_row == id
        index_addresses = self._mask_2_addresses(index_mask)
        return self.index[index_addresses, 1:] #Strips off the index metainfo

    def get_values(self, id):
        """ Given an id, returns the value entries associated"""
        index_row = self.index[:, 0]
        index_mask = index_row == id
        index_addresses = self._mask_2_addresses(index_mask)
        return self.values[index_addresses]

    def get_sparse(self, id):
        """ Given an id, returns the sparse entries associated"""
        if not hasattr(self, 'sparse'):
            self.sparse = {}
        if id not in self.sparse:
            self.sparse[id] = self._update(id)
        return self.sparse[id]

    ### Tracking methods ###

    def get_id(self):
        """ Returns a unique id for a parameter"""
        count = self.counter.item()
        self.counter += 1
        return count

    def __init__(self,
                 quantity: int,
                 capture_quantity: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 requires_grad: bool = True):
        super().__init__()

        #Create the pool
        allocated = torch.full([quantity], False)
        index = torch.empty([quantity, 3],
                                  dtype=torch.int64,
                                  device=device)
        values = torch.empty([quantity],
                                   dtype=dtype,
                                   device=device,
                                   requires_grad=requires_grad)
        self.register_buffer('allocated', allocated)
        self.register_buffer('index', index)
        self.register_parameter('values', nn.Parameter(values, requires_grad=requires_grad))

        #Create the parameter tracker and counter.
        self.register_buffer('counter', torch.tensor(0))
        self.params = nn.ModuleDict()

        #Store some values
        self.dtype = dtype
        self.device = device

class SparseParameter(nn.Module):
    """

    An interface and capture engine.

    The parameters running this are actually located somewhere on
    a parameter pool. Any resulting captures are, however, stored
    locally.

    -- fields --

    index: The indices in use
    value: The values in use
    sparse: The sparse representation.
    capture: The current capture quantities

    """
    @property
    def sparse(self):
       return self._backend.get_sparse(self._id)
    @property
    def index(self):
        return self._backend.get_index(self._id)
    @property
    def value(self):
        return self._backend.get_value(self._id)

    def __init__(self,
                 backend,
                 id,
                 shape,
                 capture):

        super().__init__()

        self._id = id
        self._backend = backend
        self._shape = shape
