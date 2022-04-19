import uuid
from typing import Optional, Callable, Final, Sequence, Union, Tuple, Dict, List

import torch
import numpy as np
import torch_sparse
from torch import nn



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

    ### Id management ###
    @property
    def id(self):
        return self._id

    ### Storage manager. ####
    @property
    def storage(self):
        """ Get the proper underlying storage. Build it if needed"""
        if not hasattr(self, '_storage'):
            #Support for saving models. Automatically rebuilds storage if
            #needed.
            with torch.no_grad():
                rowptr = self._rowptr
                col = self._col

                _, _, _, value = self._transact(get_addr=self._addr, get_values=True)
                value, _, _, _ = value

                self._storage = torch_sparse.SparseStorage(rowptr=rowptr,
                                                           col = col,
                                                           value=value,
                                                           is_sorted=True)
        #Return cached entry
        return self._storage
    @storage.setter
    def storage(self, new_storage: torch_sparse.SparseStorage):
        """ Set the storage to something else"""
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


            #Develop transaction parameters. This includes addresses to release,
            #addresses and values to set, and new values and priorities.

            release_addr = self._addr.masked_select(is_gone_b)
            set_addr = self._addr.masked_select(is_set_b)
            set_val = values.masked_select(is_set_a)
            new_values = values.masked_select(is_new_a)
            new_priority = torch.full([new_values.shape[0]], self.default)

            #Commit transaction. Store resulting address sequence

            release_status, set_status, new_status, _ = self._transact(
                                  release_addr = release_addr,
                                  set_addr = set_addr,
                                  set_values = set_val,
                                  new_values = new_values,
                                  new_priority = new_priority)

            #Unwravel transaction. Transaction may approve or deny
            #ANY portion of the attempt, including release, set
            #or new.
            #
            # Assemble, using the resulting indexmask, the final
            # index. The indexmask is a tensor of indices, meant
            # to indicate whether the value in this position
            # was successfully changed on the backend. Make
            # an index out of the successful values, and thus
            # current state.

            release_index = old_index[self._mask2index(is_gone_b)]
            set_index = index[self._mask2index(is_set_a)]
            new_index = index[self._mask2index(is_new_a)]

            release_addr, release_fail_indexmask = release_status
            set_addr, set_success_indexmask = set_status
            new_addr, new_success_indexmask = new_status

            release_index = release_index[release_fail_indexmask]
            set_index = set_index[set_success_indexmask]
            new_index = new_index[new_success_indexmask]

            final_addr = torch.concat([release_addr, set_addr, new_addr], dim=0)
            final_index = torch.concat([release_index, set_index, new_index], dim=0)

            #With the final index, and final address, completed, commit the current
            #state to instance memory.

            self._commit(final_index[:, 0], final_index[:, 1], final_addr)

    def release(self,
                addr: torch.Tensor):
        """

        Rebuilds storage with the given addresses released.

        Utilized primarily by the memory management backend
        when it needs to reclaim memory from somewhere.

        A call to this is a declaration:
        "Hey, you are not responsible for these addresses anymore"

        :param addr: A tensor, consisting of the addresses to release
        """
        with torch.no_grad():
            #Develop the current index

            row = self.storage.row()
            col = self.storage.col()

            #Figure out what addresses need to be retained, and the corrosponding indices

            broadcast_a: torch.Tensor = addr.unsqueeze(-1) #(M, 1)
            broadcast_b: torch.Tensor = self._addr.unsqueeze(0) #(1, N)
            existence_bool: torch.Tensor = (broadcast_a == broadcast_b)  #(M, N)

            released_addr = existence_bool.any(dim=0)
            retained_addr = torch.logical_not(released_addr)
            retained_indices = torch.arange(retained_addr.shape[0], device=self.device)
            retained_indices = retained_indices.masked_select(retained_addr)

            #Rebuild with only the retained quantities remaining. Create compressor,
            #then compress, then store.

            compression_row = row[retained_indices]
            compression_col = col[retained_indices]
            compression_addr = self._addr[retained_indices]

            self._commit(compression_row, compression_col, compression_addr)

    ### Priority manager ###
    @property
    def priority(self) -> torch.Tensor:
        """Return the priorities associated with the current addresses"""

        _, _, _, status = self._transact(self.id,
                                         get_addr=self._addr, get_priority=True)
        _, priority, _, _ = status

        return priority

    @priority.setter
    def priority(self, item: Union[float, torch.Tensor]):
        """ Sets the priority. May be a float, or a 1d tensor"""

        #Prep
        assert isinstance(item, (float, torch.Tensor))
        if isinstance(item, float):
            item = torch.full([self._addr.shape[0]], item, dtype=torch.float16)
        else:
            item = item.type(torch.float16)

        assert item.dim() == 1
        assert item.shape[0] == self._addr.shape[0]

        self._transact(self.id,
                       set_addr = self._addr,
                       set_priority = item)

    ### Helper functions ###
    def _mask2index(self, mask):
        """ A small function """
        assert torch.all(mask.sum(-1) == 1)
        index = torch.arange(mask.shape[-1], dtype=torch.int64).view(-1, 1)
        index = mask.matmul(index).squeeze()
        return index

    ### Helper property

    #Note that in order for distributed broadcast to work
    #tensors must be the same shape on all objects. To support this,
    #the indices are stored as a mask of memory length, and synthesized
    #into numbers on the fly.
    @property
    def _addr(self):
        """ Gets the current address"""
        return self._mask2index(self._addr_mask)
    @_addr.setter
    def _addr(self, new_address):
        address = torch.full([self._addr_mask.shape[0]], False, device=self.device)
        address[new_address] = True
        self._addr_mask = address


    ### State modifiers
    #
    # These are the ONLY functions in the class which are allowed to
    # set to anything persistant, whether it be in the backend or
    # instance variables.

    def _commit(self, row, col, addr):
        """

        Commits a particular row, col, addr sequence
        to instance fields. This is the only function allowed
        to change entries in the instance.

        :param row: the row we are dealing with
        :param col: the column entries
        :param addr: the corrosponding addresses
        """
        #Build compression fixture, and perform sort.

        sort_permutation = torch.arange(row.shape[0], device=self.device)
        sort_permutation = torch_sparse.SparseStorage(row=row,
                                                      col=col,
                                                      value=sort_permutation).value()
        final_row = row[sort_permutation]
        final_col = col[sort_permutation]
        final_addr = addr[sort_permutation]

        _, _, _, status = self._transact(self.id,
                                         get_addr=final_addr,
                                         get_values=True)
        final_value, _, _, _ = status

        # Create the final storage, and store persistent quantities

        self._storage = torch_sparse.SparseStorage(row=final_row,
                                                   col=final_col,
                                                   value=final_value,
                                                   is_sorted=True)

        self._rowptr = self._storage.rowptr()
        self._col = self._storage.col()
        self._addr = final_addr

    def _transact(self,
                 # Release transact quantities. These resolve first
                 release_addr: Optional[torch.Tensor] = None,

                 # Set transact quantities. These resove next

                 set_addr: Optional[torch.Tensor] = None,
                 set_values: Optional[torch.Tensor] = None,
                 set_priority: Optional[torch.Tensor] = None,

                 set_write_enabled: Optional[torch.Tensor] = None,
                 set_free_enabled: Optional[torch.Tensor] = None,

                 # New transact quantities. These resolve third

                 new_values: Optional[torch.Tensor] = None,
                 new_priority: Optional[torch.Tensor] = None,

                 # Fetch transact quanties. these resolve last

                 get_addr: Optional[torch.Tensor] = None,
                 get_priority: Optional[bool] = False,
                 get_values: Optional[bool] = False,
                 ) -> Tuple[Union[None, Tuple[torch.Tensor, torch.Tensor]],
                            Union[None, Tuple[torch.Tensor, torch.Tensor]],
                            Union[None, Tuple[torch.Tensor, torch.Tensor]],
                            Union[None, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """

        The primary transaction manager for the instance. Connects between
        the instance and the backend. Transactions are committed by the backend in
        one go, and resolving in the following sequence:

        1) Release
        2) Set
        3) New
        4) Get

        The transaction will return a sequence of status dictionaries properties
        when relavent to allow interfacing with the transaction result.

        This is the only method allowed to touch the backend. Generally,
        the inputs to a particular segment should either be 1D tensors of
        equal length and correct type, or None.

        :param release_addr: Addresses to releas

        :param set_addr: Addresses to set
        :param set_values: The values to set
        :param set_priority: The priority when releasing
        :param set_write_enabled: Sets whether writing is allowed
        :param set_free_enabled: Sets whether freeing is allowed.

        :param new_values: Values to get addressses for
        :param new_priority: Priorities for values

        :param get_addr: addresses to get from.
        :param get_priority: Whether to fetch priority or not.
        :param get_values: Whether to fetch values or not.
        :returns:
            release_status: How the release attempt went.
            set_status: How the set attempt went
            new_status: How the new attempt went
            get_status: The items which have been gotten
        """

        try:
            status = self._backend.transact(self._id,

                                            #Release
                                            release_addr=release_addr,

                                            #set params

                                            set_addr = set_addr,
                                            set_values = set_values,
                                            set_priority = set_priority,
                                            set_write_enabled = set_write_enabled,
                                            set_free_enabled = set_free_enabled,

                                            #new params

                                            new_values = new_values,
                                            new_priority = new_priority,

                                            #get params

                                            get_addr = get_addr,
                                            get_values = get_values,
                                            get_priority=get_priority,

                                            )
            return status
        except Exception as err:
            msg = "Error during transaction: %s" % err
            raise RuntimeError(msg) from err


    ### Initialization ###
    def __init__(self,
                 backend,
                 device: torch.device = None,
                 default_priority: float = 1.0,
                 ):
        """

        The initialization for the storage module. This must
        specify the backend, and may optionally specify the device
        and default priority.

        :param backend: A SparseParamServer
        :param device: The device to build myself on
        :param default_priority: The priority modifier. This influences whether or
            not a particular value is likely to be purged when memory needs to be
            freed.

            Higher numbers means more important and less likely to be purged.
        """
        # Start torch

        super().__init__()

        assert isinstance(backend, ParamServer)

        # Store persistent placeholders.
        self._rowptr = torch.empty([0], dtype=torch.int64, device=device)
        self._col = torch.empty([0], dtype=torch.int64, device=device)
        self._addr_mask = torch.full([backend.total], False, device=device)
        self._id = torch.Tensor(hash(uuid.uuid1()), device=device, dtype=torch.int32)

        self.register_buffer('_rowptr', self._rowptr)
        self.register_buffer('_col', self._col)
        self.register_buffer('_addr_mask', self._addr_mask)
        self.register_buffer('_id', self._id)

        #store backend and default priority
        self._backend = backend
        self.default = default_priority


class ParamServer(nn.Module):
    """



    The base class for the involved parameter server.

    This class is responsible for declaring a parameter space
    which can be served from, and handling transact queries
    from any attached StorageModules. It performs parameter
    authentication, as well as get, set, release, and new handling.

    Importantly, all subclasses of this must implement transact. Also,
    of note, violating permission rules does NOT throw an error. Instead,
    the server determined how to handle the situation, and only returns addresses
    to the items which did successfully persist or execute. Access rule violation
    however, does throw an error.


    """
    ### Helper functions ###
    def _mask2index(self, mask):
        """ A small function """
        assert torch.all(mask.sum(-1) == 1)
        index = torch.arange(mask.shape[-1], dtype=torch.int64).view(-1, 1)
        index = mask.matmul(index).squeeze()
        return index

    ### Memory information properties ###
    @property
    def total(self):
        """ The total amount of memory"""
        return self._active.shape[0]
    @property
    def free(self):
        """ The amount of free memory"""
        return self.total - self._active.sum()
    @property
    def used(self):
        """ The amount of used memory"""
        return self._active.sum()

    ### Basic query subactions ###
    def release(self, id: torch.Tensor,
                release_addr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Releases addresses, if this is permitted. ID and
        release_addr must be provided. Returned is a
        tuple, indicating first any addresses that were
        not released, and second which indexes of the incoming
        release tensor were not successfully freed.


        :param id: The id the query came from
        :param release_addr: The addresses to release. 1D float64 tensor
        :returns:
            remaining_addresses,
            remaining_indexmask
        """

        #Validation
        assert torch.is_tensor(id)
        assert id.dim() == 0
        assert id.dtype == torch.int32

        assert torch.is_tensor(release_addr)
        assert release_addr.dim() == 1
        assert release_addr.dtype == torch.int64

        assert (release_addr >= 0).all()
        assert (release_addr < self.total)

        if (self._ids[release_addr] != id).any():
            raise MemoryError("Attempt by %s to access memory it does not own" % id.item())


        #Freeing. Figure out what indices can be freed, then set those as
        #freed and reset all flags. For the ones that cannot be freed,
        #figure out what the addresses were and return them, plus the
        #indexmask

        can_release = self._free_enabled[release_addr]
        cannot_release = torch.logical_not(can_release)

        to_release = release_addr.masked_select(can_release)
        not_released = release_addr.masked_select(cannot_release)

        self._active[to_release] = False
        self._write_enabled[to_release] = False
        self._ids[to_release] = -1

        return not_released, self._mask2index(not_released)






        pass
    def _set(self, id: torch.Tensor,
            set_addr: torch.Tensor,
            set_values: Union[torch.Tensor, None],
            set_priority: Union[torch.Tensor, None],
            set_write_enabled: Union[torch.Tensor, None],
            set_free_enabled: Union[torch.Tensor, None])\
                                                ->  Tuple[List[Callable],
                                                    Tuple[torch.Tensor, torch.Tensor]]:
        """

        The set action of the backend. set_addr is required, and must be an int64
        tensor of addresses. Set value, set_priority, set_write_enabled, and
        set_read_enabled are all optional, and must be tensors of the same
        length as set_addr.

        Note that when set_write_enabled is not None, it is assumed to be the
        case that we wish to write the current entries, THEN set the access permissions.

        :param id: The id of the query caller
        :param set_addr: A 1D int64 tensor. The addresses to modify
        :param set_values: A 1D tensor, or None. The values to set
        :param set_priority: A 1D float32 tensor, or None. The associated priority
        :param set_write_enabled: A 1D bool tensor, or None. Whether value is write enabled
        :param set_free_enabled: A 1D bool tensor, or None. Whether a value is free enabled
        :return:
        """

        #Validation
        assert torch.is_tensor(id)
        assert id.dim() == 0
        assert id.dtype == torch.int32

        assert torch.is_tensor(set_addr)
        assert set_addr.dim() == 1
        assert set_addr.dtype == torch.int64
        assert (set_addr >= 0).all()
        assert (set_addr < self.total).all()

        if (self._ids[set_addr] != id).any():
            raise MemoryError("Attempt by %s to access memory it does not own" % id.item())




        #Transaction construction and verification.

        #We validate items as we go along, and construct a
        #list called transaction containing functions which need to be
        #called to perform the transaction.

        #Once all items check out, a loop applies them all.

        transaction = []

        if self._write_enabled is not None:
            assert torch.is_tensor(set_write_enabled)
            assert set_write_enabled.dim() == 1
            assert set_write_enabled.shape[0] == set_addr.shape[0]
            assert set_write_enabled.dtype == torch.bool

            def set():
                self._write_enabled[set_addr] = set_write_enabled
            transaction.append(set)

            #Overwrite enabled
            can_write_mask = torch.full([set_addr.shape[0]], True)
            can_write_addr = set_addr.masked_select(can_write_mask)
        else:
            #No overwrite. Obey access control.
            can_write_mask = self._write_enabled[set_addr]
            can_write_addr = set_addr.masked_select(can_write_mask)

        if set_values is not None:
            assert torch.is_tensor(set_values)
            assert set_values.dim() == 1
            assert set_values.shape[0] == set_addr.shape[0]
            assert set_values.dtype == self._memory.dtype

            #Reduce to the items which we can write to
            set_values = set_values.masked_select(can_write_mask)
            def set():
                self._memory[can_write_addr] = set_values
            transaction.append(set)
        if set_priority is not None:
            assert torch.is_tensor(set_priority)
            assert set_priority.dim() == 1
            assert set_priority.shape[0] == set_addr.shape[0]
            assert set_priority.dtype == torch.float32

            set_priority = set_priority.masked_select(can_write_mask)
            def set():
                self._priority[can_write_addr] = set_priority
            transaction.append(set)
        if set_free_enabled is not None:
            assert torch.is_tensor(set_free_enabled)
            assert set_free_enabled.dim() == 1
            assert set_free_enabled.shape[0] == set_addr.shape[0]
            assert set_free_enabled.dtype == torch.bool

            set_free_enabled = set_free_enabled.masked_select(can_write_mask)
            def set():
                self._free_enabled[can_write_addr] = set_free_enabled
            transaction.append(set)

        #All verification and construction complete. Return the pending
        #transaction, and the status

        return transaction, (set_addr, torch.arange(set_addr.shape[0]))





        pass
    def new(self, id, new_values, new_priority) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def get(self, id, get_addr, get_values, get_priority)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def transact(self,
                  id: torch.Tensor,

                  #Release transact quantities. These resolve first
                  release_addr: Optional[torch.Tensor] = None,

                  #Set transact quantities. These resove next

                  set_addr: Optional[torch.Tensor] = None,
                  set_values: Optional[torch.Tensor] = None,
                  set_priority: Optional[torch.Tensor] = None,

                  set_write_enabled: Optional[torch.Tensor] = None,
                  set_free_enabled: Optional[torch.Tensor] = None,


                  #New transact quantities. These resolve third

                  new_values: Optional[torch.Tensor] = None,
                  new_priority: Optional[torch.Tensor] = None,

                  #Fetch transact quanties. these resolve last

                  get_addr: Optional[torch.Tensor] = None,
                  get_priority: Optional[bool] = False,
                  get_values: Optional[bool] = False,
                 ) -> Tuple[Union[None, Tuple[torch.Tensor, torch.Tensor]],
                            Union[None, Tuple[torch.Tensor, torch.Tensor]],
                            Union[None, Tuple[torch.Tensor, torch.Tensor]],
                            Union[None, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """

        The primary transaction manager for the instance. Connects between
        the instance and the backend. Transactions are committed by the backend in
        one go, and resolving in the following sequence:

        1) Release
        2) Set
        3) New
        4) Get

        The transaction will return a sequence of status dictionaries properties
        when relavent to allow interfacing with the transaction result.

        This is the only method allowed to touch the backend. Generally,
        the inputs to a particular segment should either be 1D tensors of
        equal length and correct type, or None.

        :param release_addr: Addresses to releas

        :param set_addr: Addresses to set
        :param set_values: The values to set
        :param set_priority: The priority when releasing
        :param set_write_enabled: Sets whether writing is allowed
        :param set_free_enabled: Sets whether freeing is allowed.

        :param new_values: Values to get addressses for
        :param new_priority: Priorities for values
        :param get_addr: addresses to get from.
        :param get_priority: Whether to fetch priority or not.
        :param get_values: Whether to fetch values or not.
        :returns:
            release_status: How the release attempt went.
            set_status: How the set attempt went
            new_status: How the new attempt went
            get_status: The items which have been gotten
        """

        release_status = None
        set_status = None
        new_status = None
        get_status = None

        if release_addr is not None:
            release_status = self.release(id, release_addr)
        if set_addr is not None:
            set_status = self.set(id, set_addr, set_values, set_priority, set_write_enabled, set_free_enabled)
        if new_values is not None:
            new_status = self.new(id, new_values, new_priority)
        if get_status is not None:
            get_status = self.get(id, get_addr, get_values, get_priority)

        return release_status, set_status, new_status, get_status
    def __init__(self,
                 quantity: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 requires_grad: bool = True,

                 reclamation_activation: Optional[Callable] = None

                 ):
        super().__init__()

        if reclamation_activation is None:
            reclamation_activation = torch.abs
        else:
            # todo: Figure out how to save a function. As it stands, will not save when not none
            raise NotImplementedError("This features is not yet available")

        #Create permission and activity trackers. Follow these
        #up with memory change flags. Then register these as buffers

        self._active = torch.full([quantity], False, device=device)
        self._ids = -torch.ones([quantity], dtype=torch.int32)

        self._free_enabled = torch.full([quantity], True, device=device)
        self._write_enabled = torch.full([quantity], True, device=device)

        self.register_buffer('_active', self._active)
        self.register_buffer('_ids', self._ids)
        self.register_buffer('_free_enabled', self._free_enabled)
        self.register_buffer('_write_enabled', self._write_enabled)


        #Create parameters memory, priority
        self._memory = torch.empty([quantity],
                                   dtype=dtype,
                                   device=device,
                                   requires_grad=requires_grad)
        self._priority = torch.empty([quantity],
                                     dtype=torch.float16,
                                     device=device,
                                     requires_grad=False)

        self._memory = nn.Parameter(self._memory, requires_grad=requires_grad)
        self._priority = nn.Parameter(self._priority, requires_grad=False)

        #Store persistant
        self._dtype = dtype




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
