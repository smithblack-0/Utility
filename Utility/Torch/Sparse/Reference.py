import torch
from torch import nn
from torch_sparse import SparseStorage


class Reference(nn.Module):
    """
    An interface to a memory backend.
    """
    ### Static ###
    @property
    def storage(self):
        """ The stored sparse representation. Returned!"""
        if not hasattr(self, '_cached_storage') or self.version != self._cache_version:
            row = self.index[:, 0]
            col = self.index[:, 1]
            value = self.value
            self._cache_storage = SparseStorage(row = row,
                                        col=col,
                                        value=value,
                                        is_sorted=True,
                                        trust_data=True)
            self._cache_version = self._latest_version.clone()
        return self._cache_storage


    @property
    def version(self):
        """The modification version. Used to tell when to rebuild"""
        return self._latest_version
    @property
    def reference(self):
        """ A lookup to the underlying bool reference"""
        id = self._id.to(self._backend.device)
        return self._backend.reference[:, id].to(self.device)
    @reference.setter
    def reference(self, value):
        """ A syncronization object to the underlying bool reference"""
        id = self._id.to(self._backend.device)
        value = value.to(self._backend.device)
        self._backend.reference[:, id] = value

    @property
    def addresses(self):
        """ The addresses referred to by the reference."""
        addresses = torch.arange(self._backend.total, dtype=torch.int32, device=self.device)
        addresses = addresses.masked_select(self.reference)
        return addresses
    @property
    def _addresses(self):
        """ addresses, but on the backend device. Internal"""
        addresses = torch.arange(self._backend.total, dtype=torch.int32, device=self.device)
        addresses = addresses.masked_select(self.reference).to(self._backend.device)
        return addresses

    ### Manipulatebles ###
    @property
    def index(self):
        """ Get the current index"""
        return self._backend.index[self._addresses].to(self.device)
    @property
    def priority(self):
        """ Get the current priority"""
        return self._backend.priority[self._addresses].to(self.device)
    @property
    def value(self):
        """ Get the current value """
        return self._backend.value[self._addresses].to(self.device)
    @index.setter
    def index(self, new_index):
        """ Set the current index"""
        assert torch.is_tensor(new_index)
        assert new_index.dim() == 2
        assert new_index.shape[0] == self.index.shape[0]
        assert new_index.dtype == self.index.dtype

        with torch.no_grad():
            self._backend.index[self._addresses] = new_index.to(self._backend.device)
            self.tick()
    @priority.setter
    def priority(self, new_priority):
        """ Set the current priority"""
        assert torch.is_tensor(new_priority)
        assert new_priority.dim() == 1
        assert new_priority.shape[0] == self.priority.shape[0]
        assert new_priority.dtype == self.priority.dtype

        with torch.no_grad():
            self._backend.priority[self._addresses] = new_priority.to(self._backend.device)
            self.tick()
    @value.setter
    def value(self, new_value):
        """ Set the current value"""
        assert torch.is_tensor(new_value)
        assert new_value.dim() == 1
        assert new_value.shape[0] == self.value.shape[0]
        assert new_value.dtype == self.value.dtype

        with torch.no_grad():
            self._backend.value[self._addresses] = new_value.to(self._backend.device)
            self.tick()

    #Functions
    def release(self):
        """

        Release the currently assigned memory block

        """
        with torch.no_grad():

            fill = torch.full([self._backend.length], False, device=self._backend.device)
            self.reference = fill
            self.tick()

    def reserve(self,
                index: torch.Tensor,
                value: torch.Tensor,
                priority: torch.Tensor):
        """

        Request a reservation for the given
        index, value, priority sequence.

        :param index: The int64 index tensor
        :param value: The dtype 1D value tensor
        :param priority: The dtype 1D priority tensor
        :return: None
        """

        assert torch.is_tensor(index)
        assert torch.is_tensor(value)
        assert torch.is_tensor(priority)

        assert index.dim() == 2
        assert value.dim() == 1
        assert priority.dim() == 1

        length = index.shape[0]
        assert value.shape[0] == length
        assert priority.shape[0] == length

        assert index.shape[1] == 2

        assert index.dtype == self._backend.index.dtype
        assert value.dtype == self._backend.value.dtype
        assert priority.dtype == self._backend.priority.dtype

        with torch.no_grad():
            if self._reference is not None:
                self.release()

            id = self._id.to(self._backend.device)
            index = index.to(self._backend.device)
            value = value.to(self._backend.device)
            priority = priority.to(self._backend.device)

            self._backend.reserve(id, index, value, priority).to(self.device)
            self.tick()

    def tick(self):
        """ Increases the version count by 1"""
        self._latest_version += 1
    def __init__(self,
                 backend,
                 device=None):
        super().__init__()

        self._id = backend.register(self)
        self._latest_version = torch.Tensor(0, dtype=torch.int64, device=device)
        self._cache_version = torch.Tensor(0, dtype=torch.int64, device=device)

        self.register_buffer('_id', self._id)
        self.register_buffer('_latest_version', self._id)
        self._backend = backend

