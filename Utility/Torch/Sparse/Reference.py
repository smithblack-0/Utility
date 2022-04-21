import torch
from torch import nn


class Reference(nn.Module):
    """
    An interface to a memory backend.

    With the exception of the id for this memory access point, nothing
    is actually stored on this object. Instead, it acts as a clear
    passthrough to the assigned memory region, allowing the reservation
    and setting of values, index, and priority parameters, along
    with verison information

    Features include the ability to set index, value, and priority information
    directly from the instance, the ability to release the current allocation,
    and the ability to attempt to set an allocation.

    The word "attempt" bears a little explanation. It is the case that the
    reference hands off the allocation to the memory backend. It is up to
    the backend to decide whether or not it will set aside memory. It may
    decide not to if, for example, there is not enough remaining and the
    magnitude is too low.
    """
    ### Static ###
    @property
    def version(self):
        """The modification version. Used to tell when to rebuild"""
        id = self._id.to(self._backend.device)
        return self._backend.latest_version[id]
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

            id = self._id.to(self._backend.device)
            self._backend.release(id)
            self.tick()
    def set(self,
            index: torch.Tensor,
            value: torch.Tensor,
            priority: torch.Tensor
            ):
      """
      Request to set the memory buffer to match the provided
      pattern. Release everything first.

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
          if self.reference.any() is not False:
              self.release()

          id = self._id.to(self._backend.device)
          index = index.to(self._backend.device)
          value = value.to(self._backend.device)
          priority = priority.to(self._backend.device)

          self._backend.reserve(id, index, value, priority)
          self.tick()


    def reserve(self,
                index: torch.Tensor,
                value: torch.Tensor,
                priority: torch.Tensor):
        """

        Request a reservation for the given
        index, value, priority sequence.

        Does NOT release the current memory before executing. Instead,
        requests new memory for all the entries.

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
            id = self._id.to(self._backend.device)
            index = index.to(self._backend.device)
            value = value.to(self._backend.device)
            priority = priority.to(self._backend.device)

            self._backend.reserve(id, index, value, priority)
            self.tick()

    def tick(self):
        """ Increases the version count by 1"""
        id = self._id.to(self._backend.device)
        self._backend.latest_version[id] += 1

    def __init__(self,
                 backend,
                 device=None):
        super().__init__()

        id = backend.register(self).to(device)
        self.register_buffer('_id', id)
        self._backend = backend

