import uuid
from typing import Optional, Callable, Final, Sequence, Union, Tuple, Dict, List

import torch
import numpy as np
import torch_sparse
from torch import nn

class TriggerModule(nn.Module):
    """

    A module capable of telling when
    a particular item has been changed
    by a distributed broadcast, syncronous
    or not.

    When a hook is registered, it watches
    to see if the
    """



class ArchStorageModule(nn.Module):
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
                addr: torch.Tensor) -> Callable:
        """

       Returns a promise to rebuild the storage with any
       given addresses eliminated once the return is called.

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

            #Return the promise

            def promise():
                self._commit(compression_row, compression_col, compression_addr)
            return promise

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
        self._reference = torch.full([backend.total], False)
        self._id = torch.Tensor(hash(uuid.uuid1()), device=device, dtype=torch.int32)

        self.register_buffer('_reference', self._reference)
        self.register_buffer('_id', self._id)

        #store backend and default priority
        self._backend = backend
        self.default = default_priority




class Storage(nn.Module):
    ### properties ###
    @property
    def addresses(self):
        return self._addresses
    @property
    def updates(self):
        return self._updates
    @property
    def owner(self):
        return self._owner
    @property
    def index(self):
        return self._index
    @property
    def priority(self):
        return self._priority
    @property
    def value(self):
        return self._value
    @property
    def transfer(self):
        return self._transfer
    @property
    def length(self):
        return self._owner.shape[-1]

    @owner.setter
    def owner(self, value):
        """ Changing owners logic"""

        assert torch.is_tensor(value)
        assert value.shape == self._owner.shape
        assert value.dtype == torch.bool
        with torch.no_grad():
            self._owner.copy_(value)
    @index.setter
    def index(self, index):
        """ Changing index logic"""
        assert torch.is_tensor(index)
        assert self._index.shape == index.shape
        assert self._index.dtype == index.dtype
        with torch.no_grad():
            self._index.copy_(index)
    @priority.setter
    def priority(self, priority):
        assert torch.is_tensor(priority)
        assert self._priority.shape == priority.shape
        assert self._priority.dtype == priority.dtype
        with torch.no_grad():
            self._priority.copy_(priority)
    @value.setter
    def value(self, value):
        assert torch.is_tensor(value)
        assert self._value.shape == value.shape
        assert self._value.dtype == value.dtype
        with torch.no_grad():
            self._value.copy_(value)
    @transfer.setter
    def transfer(self, transfer):
        assert torch.is_tensor(transfer)
        assert self._transfer.shape == transfer.shape
        assert self._transfer.dtype == transfer.dtype
        with torch.no_grad():
            self._transfer.copy_(transfer)
    @updates.setter
    def updates(self, value):
        self._updates = value
    ### Inferred ###

    @property
    def total(self):
        return self._owner.shape[0]
    @property
    def used(self):
        return self.owner.any(dim=-1).sum()
    @property
    def free(self):
        return self.total - self.used


    ### Helper ###
    def _mask2index(self, mask):
        index = torch.arange(mask.shape[-1], device=mask.device, dtype=torch.int64)
        index = index.masked_select(mask)
        return index

    #Functions
    def copy(self):
        """ Creates a copy of my storage """
        item = Storage(self.owner,
                       self.index,
                       self.priority,
                       self.value,
                       self.transfer)
        item.updates = self.updates
        return item
    def from_copy(self, copy):
        """ Updates my storage from a copy"""
        self.updates = copy.updates
        self.owner.copy_(copy.owner)
        self.index.copy_(copy.index)
        self.priority.copy_(copy.priority)
        self.value.copy_(copy.value)
        self.transfer.copy_(copy.transfer)
        return self

    def expand(self,
               amount: int):
        new_owner = torch.full([self.length, amount], False)
        new_owner = torch.concat([self.owner, new_owner], dim=-1)
        self._owner = new_owner

    def __init__(self,
                 owner: torch.Tensor,
                 index: torch.Tensor,
                 priority: torch.Tensor,
                 value: torch.nn.Parameter,
                 transfer: torch.Tensor,
                 ):
        super().__init__()

        #Run validation
        assert torch.is_tensor(owner)
        assert torch.is_tensor(index)
        assert torch.is_tensor(priority)
        assert torch.is_tensor(value)
        assert torch.is_tensor(transfer)

        assert owner.dim() == 2
        assert index.dim() == 2
        assert priority.dim() == 1
        assert value.dim() == 1
        assert transfer.dim == 1

        length = owner.shape[0]
        assert index.shape[0] == length
        assert priority.shape[0] == length
        assert value.shape[0] == length
        assert transfer.shape[0] == length

        assert owner.dtype == torch.bool
        assert index.dtype == torch.int64
        assert value.dtype == priority.dtype
        assert value.dtype == transfer.dtype

        device =  owner.device
        assert device == index.device
        assert device == value.device
        assert device == priority.device
        assert device == transfer.device

        assert isinstance(value, nn.Parameter)

        #Store items

        self._updates = torch.empty([0], dtype=torch.int64, device=device)
        self._addresses = torch.arange(length, device =device, dtype=torch.int64)
        self._owner = owner
        self._index = index
        self._priority = priority
        self._transfer = transfer
        self._value = value

        self.register_buffer('_addresses', self._addresses)
        self.register_buffer('_owner', self._owner)
        self.register_buffer('_index', self._index)
        self.register_buffer('_priority', self._priority)
        self.register_buffer('_transfer', self._transfer)
        self.register_parameter('_value', self._value)



class ParamMan(nn.Module):
    """
    The storage location for parameters.
    The interface through which such things can be modified.

    """



    class idStorage(nn.Module):
        """ A microclass for holding ids"""
        def __init__(self):
            super().__init__()
    ### Basic query subactions ###
    def _get(self, environment: Dict,
            id: torch.Tensor,
            get_addr: torch.Tensor,
            get_values: bool,
            get_priority: bool)-> Tuple[Dict,
                                        Tuple[Union[None, torch.Tensor],
                                              Union[None, torch.Tensor]]
                                        ]:
        """

        The get action


        :param environment:
        :param id:
        :param get_addr:
        :param get_values:
        :param get_priority:
        :return:
        """
        #TODO: Impliment
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
        2) New
        3) Set
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

        #This could use a little explanation.
        #
        #To avoid any corruption, we create a copy of the entire
        #runtime environment and edit that. Once we are sure that
        #we have not made any mistakes, we go ahead and copy the results
        #into our actual backend.
        environment = self._storage.copy()

        release_status = None
        set_status = None
        new_status = None
        get_status = None

        if release_addr is not None:
            environment = self._release(environment, id, release_addr)
        if new_values is not None:
            simulated_environment = self._new(environment, id, new_values, new_priority)
        if set_addr is not None:
            simulated_environment, set_status = self._set(environment, id, set_addr, set_values, set_priority, set_write_enabled, set_free_enabled)
        if get_status is not None:
            simulated_environment, get_status = self._get(environment, id, get_addr, get_values, get_priority)

        #TODO: Update so that this stores the environment, THEN returns
        return release_status, set_status, new_status, get_status

    @staticmethod
    def _set_updates(storage: Storage,
                     addresses: torch.Tensor):
        """ Sets anything which accesses the addresses
        as having experienced an update"""

        owners_experiencing_updates = storage.owner[addresses] #(N, L)
        possible_owners = torch.arange(owners_experiencing_updates.shape[1], dtype=torch.int64, device=storage.device).view(-1, 1) #(L, 1)
        updating_owners = owners_experiencing_updates.matmul(possible_owners)
        updating_owners = torch.concat([updating_owners, storage.updates])
        updating_owners = updating_owners.unique()
        storage.updates = updating_owners
        return storage




    @staticmethod
    def _release(storage: Storage,
                 addresses: torch.Tensor):
        storage.owner[addresses, :] = False
        storage.priority[addresses] = 0
        return storage

    @staticmethod
    def _set(storage: Storage,
             addresses: torch.Tensor,
             value: Union[torch.Tensor, None],
             priority: Union[torch.Tensor, None]
             ):

        if value is not None:
            storage.value[addresses] = value
        if priority is not None:
            storage.priority[addresses] = priority
        return storage

    @classmethod
    def _new(cls, storage: Storage,
             id: torch.Tensor,
             value: torch.Tensor,
             index: torch.Tensor,
             priority: torch.Tensor):

        if storage.free < value.shape[0]:
            #Not enough memory left. Go into memory freeing routine.

            source_master = torch.full(storage.length, True, device=storage.device)
            source_new = torch.full(value.shape[0], False, device=storage.device)
            new_addresses = torch.arange(value.shape[0], dtype=torch.int64, device=storage.device)

            sources = torch.concat([source_master, source_new])
            net_addresses = torch.concat([storage.addresses, new_addresses])
            net_value = torch.concat([storage.value, value], dim=-1)
            net_priority = torch.concat([storage.priority, priority], dim=-1)
            net_score = net_value*net_priority

            #TODO: Replace with activation
            net_score = torch.abs(net_score)

            permuter = torch.argsort(net_score, descending=True)
            final_addresses = net_addresses[permuter]
            final_sources = sources[permuter]

            retained_addresses = final_addresses[:storage.length]
            released_addresses = final_addresses[storage.length:]

            retained_sources = torch.logical_not(final_sources[:storage.length])
            released_sources = final_sources[storage.length:]

            master_addresses_needing_release =  released_addresses.masked_select(released_sources)
            item_addresses_getting_set = retained_addresses.masked_select(retained_sources)

            storage = cls._release(storage, master_addresses_needing_release)
            index = index[item_addresses_getting_set]
            value = value[item_addresses_getting_set]
            priority = priority[item_addresses_getting_set


    def set(self,
            id: torch.Tensor,
            index: torch.Tensor,
            value: torch.Tensor,
            priority: torch.Tensor,

            )-> Union[Exception, None]:

        #Calculate required operations. release, set, new.

        master_addresses = self._storage.addresses.masked_select(self._storage.owner[:, id])
        item_addresses = torch.arange(index.shape[0], dtype=torch.int64, device=self.device)
        master_index = self._storage.index[master_addresses, :]

        broadcast_master = master_index.unsqueeze(0) #(1, new, 2)
        broadcast_item = index.unsqueeze(1), #(current, 1, 2)
        exchange_bool = (broadcast_item == broadcast_master).all(dim=-1) #(new, current)

        #Generate masks

        is_set_master_mask = exchange_bool.any(dim=0)
        is_set_item_mask = exchange_bool.any(dim=1)

        is_discarded_mask = torch.logical_not(is_set_master_mask)
        is_new_mask = torch.logical_not(is_set_item_mask)

        #Generate addresses. Both global, and local.

        set_master_addresses = master_addresses.masked_select(is_set_master_mask)
        discarded_master_addresses = master_addresses.masked_select(is_discarded_mask)

        set_item_addresses = item_addresses.masked_select(is_set_item_mask)
        new_item_addresses = item_addresses.masked_select(is_new_mask)

        #Generate set and new subsections

        set_value = None
        new_value = None

        set_priority = None
        new_priority = None

        if value is not None:
            set_value = value[set_item_addresses]
            new_value = value[new_item_addresses]
        if priority is not None:
            set_priority = priority[set_item_addresses]
            new_priority = priority[new_item_addresses]

        new_index = index[new_item_addresses]

        #Use my now constructed addresses to perform updates. Do first the
        #release update, then the set update, then the new update.

        environment = self._storage.copy()
        environment = self._release(environment, discarded_master_addresses)
        environment = self._set(environment, set_master_addresses, set_value, set_priority)
        environment = self._new(environment, new_value, new_index, new_priority)

        modified = environment.
        self._storage.from_copy(environment)









        #Divide the items into sections requiring release, set, and new generation

        release_addr = master_addresses.masked_select(is_discarded_mask)
        set_addr = master_addresses.masked_select(is_set_current_mask)

        set_new_addr =




        environment = self._storage.copy()

        owned_addresses = torch.arange(self.total, dtype=torch.int64, device=self.device)
        owned_addresses = owned_addresses.masked_select(environment.owner[:, id])

        owned_index = environment.index[owned_addresses, :]
        owned_value = environment.value[owned_addresses]
        owned_priority = environment.priority[owned_addresses]







    def __init__(self,
                 quantity: int,
                 device,
                 dtype,
                 requires_grad=True):
        super().__init__()

        #Ownership, priority, and index setup, plus transfer buffer
        owner = torch.empty([quantity, 0], dtype=torch.bool, device=device)
        index = torch.empty([quantity, 2], dtype=torch.int64, device=device)
        priority = torch.full([quantity], 0, dtype=torch.float16, device=device)
        transfer = torch.empty([quantity], dtype=dtype, device=device)

        value = torch.empty([quantity], dtype=dtype, requires_grad=requires_grad, device=device)
        value = nn.Parameter(value, requires_grad=requires_grad)

        self._storage = Storage(owner, index, priority, value, transfer)

        #Store static
        self._dtype = dtype
        self._requires_grad = requires_grad





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
            # TODO: Figure out how to save a function. As it stands, will not save when not none
            raise NotImplementedError("This features is not yet available")

        #Create permission and activity trackers. Follow these
        #up with memory change flags. Then register these as buffers






        #Create memory and module storage
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
        self._module_storage = nn.ModuleList()

        #Store persistant
        self._dtype = dtype

        # TODO: This will not stay around in a saveload cycle. Fix
        self._reclamation_activation = reclamation_activation

        #Store DistributedParallel idflag. If this does NOT eq

