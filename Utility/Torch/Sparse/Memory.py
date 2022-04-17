from typing import Optional, Callable, Final, Sequence, Union

import torch
import numpy as np
import torch_sparse
from torch import nn


class COOParamMemory(nn.Module):
    """

    Creates a new memory bank for a sparse-dynamic environment based on the
    provided parameters. This is designed to enable transparent interfacing with
    existing optim frameworks. The instance exposes a __setitem__ and __getitem__ method
    which allows the assigning of COO coordinates to the memory backend. If coordinates
    and parameters are later removed,  or added, since the underlying parameters remain
    around torch optim's remain happy.

    For best performance, only trim parameter which have minimal contribution to the
    model, to prevent complications due to optim persistant values. Also,

    -- methods --

    __getattr__:
    __setattr__:


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

        # Create the index found boolean matrix
        self_index = self.index  # (M, 3)
        self_index.unsqueeze(0)  # (1, M, 3)
        index_broadcast = index.unsqueeze(1)  # (N, 1, 3)
        index_found_bool = (self_index == index_broadcast).all(dim=-1)  # (N, M)
        addresses = self._mask_2_addresses(index_found_bool)  # (N, 1)
        addresses = addresses.squeeze()

        return addresses

    def _get_partition(self, index):
        """
        Returns a function which will split any incoming
        tensorlike into regions of in, or not in, memory
        """

        # Create the index found boolean matrix
        self_index = self._index  # (M, 3)
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

        # Get length inactive indices, and find the corrosponding addresses to activate
        length = index.shape[0]
        inactive = self._allocated == False
        addresses = self._mask_2_addresses(inactive)
        if addresses.shape[0] < length:
            raise MemoryError("Amount of remaining parameter memory is less than size of allocation")
        addresses = addresses[:length]

        # Go set these as active, and return
        self._allocated[addresses] = True
        self._index[addresses] = index[:]

        return addresses

    def _release_addresses(self, addresses):
        """
        Simply releases a sequence of held addresses.
        """

        self._allocated[addresses] = False

    def _set_addresses(self, index, value):
        """

        sets the addresses in memory corrosponding to a particular index -value sequence

        if value is none, frees that memory.
        If index entries are present that have not been allocated before, allocate them
        If index entries are present that have been allocated, set them.
        """

        # Get address locations for currently located entries, along with anything not in memory
        with torch.no_grad():
            partition = self._get_partition(index)
            found_index, missing_index = partition(index)

            # Release memory and terminate if we are freeing memory.
            if value is None:
                # Find the entries that exist. Then free them
                found_addresses = self._get_addresses(found_index)
                self._release_addresses(found_addresses)
                return None

            # We are setting memory. We need to handle the cases in which the
            # amount remaining is insufficient, in which the memory is not
            # allocated, and in which it is.

            # Handle throw in case of not enough remaining memory
            if self.overflow is False and missing_index.shape[0] > self.free:
                raise MemoryError("Insufficient memory remaining to store index")

            # Store values which have locations already assigned
            found_values, missing_values = partition(value)
            found_addresses = self._get_addresses(found_index)
            self._values[found_addresses] = found_values

            # Handle case in which insufficient memory remains, and we
            # are not
            if missing_index.shape[0] > self.free:

                # Insufficient memory exists. Enter sort-discard mode.
                values = torch.concat([self.values, missing_values])
                index = torch.concat([self.index, missing_index])
                activation = self.activation(values)
                sort_indices = torch.argsort(activation)

                values = values[sort_indices]
                index = index[sort_indices]

                values = values[:self.length]
                index = index[:self.length]

                self._set_addresses(index, values)

            else:

                found_value, missing_value = partition(value)
                found_addresses = self._get_addresses(found_index)
                new_addresses = self._allocate_index(missing_index)

                # Make final dispatch tensor
                final_index = torch.concat([found_index, missing_index], dim=0)
                final_addresses = torch.concat([found_addresses, new_addresses], dim=0)
                final_values = torch.concat([found_value, missing_value], dim=0)

                # Store in memory

                self._index[final_addresses] = final_index
                self._values[final_addresses] = final_values

    ### Properties ###
    @property
    def free(self):
        return torch.logical_not(self._allocated).sum()

    @property
    def used(self):
        return self._allocated.sum()

    ### External interface methods ###
    def __setitem__(self, index, value, trust_input=False):
        """
        Accepts a COO index, and a tensor of values or none.
        Sets memory to produce values when the appropriate
        index is queried.

        :param index: A COO index matching the shape of the backend and contained
            within memory
        :param value: The values to set. Must be a tensor of length index, or None. If
            None, the memory will be freed.
        :raises KeyError: If the index attempts to free values which are not in memory
        :raises MemoryError: If the index attempts to set more value than there is memory for,
            and triage is not active.
        """

        #Validation
        if not trust_input:
            assert torch.is_tensor(index)
            assert index.dim() == 2
            assert index.shape[1] == self.index.shape[1]
            assert torch.is_tensor(value) or value is None

            if value is not None:
                assert value.dim() == 1
                assert value.shape[0] == index.shape[0]

            if value is None:
                partition = self._get_partition(index)
                in_memory, not_in_memory = partition(index)
                if not_in_memory.shape[0] > 0:
                    raise KeyError("Attempt to free indices which are not in memory")
        #Setting
        self._set_addresses(index, value)

    def __getitem__(self, index):
        """

        Accepts a COO index, and a tensor of values or None
        Sets the values at index to the items in values
        If values was None, instead frees the memory at index

        :param index: A COO tensor which is found entirely in memory
        :return: The values corrosponding to the COO tensor
        :raises: KeyError: If the COO index had values not in memory
        """

        assert torch.is_tensor(index)
        assert index.dim() == 2
        assert index.shape[1] == self.index.shape[1], "Shapes of instance and provide index were not the same"

        partition = self._partition(index)
        valid_index, invalid_index = partition(index)
        if invalid_index.shape[0] > 0:
            raise KeyError("Index provided with entries which do not exist", invalid_index)

        addresses = self._get_addresses(valid_index)
        return self.values[addresses]

    def __init__(self,
                 quantity: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 trim_overflow: Optional[bool] = None,
                 overflow_activation: Optional[Callable] = None,
                 requires_grad: bool = True):

        # Start torch
        super().__init__()

        # Handle overflow defaults
        if trim_overflow is None:
            trim_overflow = True
        if overflow_activation is None:
            overflow_activation = torch.abs

        # Store overflow parameters

        self.overflow = trim_overflow
        self.activation = overflow_activation

        # Create the pool
        allocated = torch.full([quantity], False)
        index = torch.empty([quantity, 3],
                            dtype=torch.int64,
                            device=device)
        values = torch.empty([quantity],
                             dtype=dtype,
                             device=device,
                             requires_grad=requires_grad)
        self.register_buffer('_allocated', allocated)
        self.register_buffer('_index', index)
        self.register_parameter('_values', nn.Parameter(values, requires_grad=requires_grad))

        # Store some values
        self.dtype = dtype
        self.device = device


class StorageModule(nn.Module):
    """
    A module capable of storing and managing
    access to a SparseStorage backend from
    torch_sparse while interfacing with
    a parameter-memory context in a torch
    state_dict compatible manner.

    -- fields --
    storage:
        A location which is intended to hold some sort
        of torch_sparse SparseStorage. May be set to, or
        retrieved.

        IMPORTANT NOTE:

        It is up to the backend to decide whether or not
        new values get memory. If the backend is full, or configured
        strangely, setting to storage may not transfer the entire
        unit into memory.
    """

    ### Storage manager. ####
    @property
    def storage(self):
        if not hasattr(self, '_storage'):
            #Support for saving models. Automatically rebuilds storage if
            #needed.

            rowptr = self._rowptr
            col = self._col
            value = self.backend[self._addr]

            self._storage = torch_sparse.SparseStorage(rowptr=rowptr,
                                                       col = col,
                                                       value=value,
                                                       is_sorted=True)
        #Return cached entry
        return self._storage
    @storage.setter
    def storage(self, new_storage: torch_sparse.SparseStorage):
        with torch.no_grad():
            #Get new index, new values

            row = new_storage.row()
            col = new_storage.col()

            index = torch.stack([row, col], dim=-1)
            values = new_storage.value()

            #Get old index.
            row = self.storage.row()
            col = self.storage.col()
            old_index = torch.stack([row, col], dim=-1)

            #Cross reference between the old and new index.
            #Figure out what indices are new, what indices
            #will be set to, and what indices will be shut off

            broadcast_a = index.unsqueeze(-3), #(1, new_items, 2)
            broadcast_b = old_index.unsqueeze(-2) #(old_items, 1, 2)
            existance_bool = (broadcast_a == broadcast_b).all(dim=-1) #(old_items, new_items)

            is_set_a = existance_bool.any(dim=0)
            is_new_a = torch.logical_not(is_set_a)

            is_set_b = existance_bool.any(dim=-1)
            is_gone_b = torch.logical_not(is_set_b)

            #Release the unused memory, then set the values on the retained common portions.

            self.backend[self._addr.masked_select(is_gone_b)] = None
            self.backend[self._addr.masked_select(is_set_b)] = values.masked_select(is_set_a)

            #Request from the backend to provided addresses for the indicated values.
            #It is up to the backend to decide which values will be allowed new addresses
            #It will return the addresses, and a function which maps to the active addresses.
            #Then, make the CSR compression features


            new_values = values.masked_select(is_new_a)
            if new_values.shape[0] > 0:
                map_to_addresses, addresses  = self.backend.request_addresses(self, new_values)
                self.backend[addresses] = map_to_addresses[new_values] #Store

            #Generate commit statements
            compression_index = index
            compression_address = self._addr.masked_select(is_set_b)

            if new_values.shape[0] > 0:
                compression_index = torch.concat([compression_index, map_to_addresses(index)], dim=0)
                compression_address = torch.concat([compression_address, addresses])

            self.commit(compression_index[:, 0], compression_index[:, 1], compression_address)
    def commit(self, row, col, addr):
        """

        Commits a particular row, col, addr sequence
        to memory. This is the only function allowed
        to change entries in the instance.

        :param row: the row we are dealing with
        :param col: the column entries
        :param addr: the corrosponding addresses
        :return:
        """
        #Build compression fixture, and perform sort.

        sort_permutation = torch.arange(row.shape[0], device=self.device)
        sort_permutation = torch_sparse.SparseStorage(row=row,
                                                      col=col,
                                                      value=sort_permutation).value()
        final_row = row[sort_permutation]
        final_col = col[sort_permutation]
        final_addr = addr[sort_permutation]
        final_value = self.backend[final_addr]

        # Create the final storage, and store persistent quantities

        self._storage = torch_sparse.SparseStorage(row=final_row,
                                                   col=final_col,
                                                   value=final_value,
                                                   is_sorted=True)

        self._rowptr = self._storage.rowptr()
        self._col = self._storage.col()
        self._addr = final_addr


    def release(self,
                addr: torch.Tensor):
        """

        Rebuilds storage with the given addresses released.

        Utilized primarily by the memory management backend
        when it needs to reclaim memory from somewhere. Items which
        are not found are not released.

        :param addr: A tensor, consisting of the addresses to release
        """
        with torch.no_grad():
            #Develop the current index

            row = self.storage.row()
            col = self.storage.col()

            #Figure out what addresses need to be retained, and the corrosponding indices

            broadcast_a = addr.unsqueeze(-1) #(M, 1)
            broadcast_b = self._addr.unsqueeze(0) #(1, N)
            existence_bool = broadcast_a == broadcast_b  #(M, N)

            released_addr = existence_bool.any(dim=0)
            retained_addr = torch.logical_not(released_addr)
            retained_indices = torch.arange(retained_addr.shape[0], device=self.device)
            retained_indices = retained_indices.masked_select(retained_addr)

            #Rebuild with only the retained quantities remaining. Create compressor,
            #then compress, then store.

            compression_row = row[retained_indices]
            compression_col = col[retained_indices]
            compression_addr = self._addr[retained_indices]

            self.commit(compression_row, compression_col, compression_addr)



    ### Initialization ###
    def __init__(self,
                 backend,
                 device = None,
                 ):
        # Start torch

        super().__init__()

        #Store persistent placeholders.

        rowptr = torch.empty([0], dtype=torch.int64, device=device)
        col = torch.empty([0], dtype=torch.int64, device=device)
        addr = torch.empty([0], dtype=torch.int64, device=device)

        self.register_buffer('_rowptr', rowptr)
        self.register_buffer('_col', col)
        self.register_buffer('_addr', addr)

        #Register and store backend
        self.backend = backend
        self.backend.register(self)





class CSRParamMemory(nn.Module):
    """
    Create a new common parameter memory backend for any number of CSR
    SparseParameters.
    """
    def __getitem__(self, item):



    def __init__(self,
                 quantity: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 throw_on_overflow: Optional[bool] = None,
                 overflow_activation: Optional[Callable] = None,
                 requires_grad: bool = True):

        # Start torch
        super().__init__()

        # Handle overflow defaults
        if trim_overflow is None:
            trim_overflow = True
        if overflow_activation is None:
            overflow_activation = torch.abs

        # Store overflow parameters

        self.overflow = trim_overflow
        self.activation = overflow_activation

        # Create the pool, storage, and addressing space.
        allocated = torch.full([quantity], False)
        values = torch.empty([quantity],
                             dtype=dtype,
                             device=device,
                             requires_grad=requires_grad)

        storage = nn.ModuleList()

        self.register_buffer('_allocated', allocated)
        self.register_parameter('_values', nn.Parameter(values, requires_grad=requires_grad))
        self.storage = storage
