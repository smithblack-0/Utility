import torch
from torch import nn
from Reference import Reference

class MemoryBackend(nn.Module):
    """
    The memory backend for a particular SparseParameter
    server.
    --- Methods ---

    register: Set aside an ID and region for the parameter interface.
    reserve: Attempt to reserve memory for the indicated features
    release: Release any memory bound to a particular id

    --- Fields ---

    reference: A 2D bool array, of length id's, quantity, tracking who owns what memory
    latest_version: A 1D int array. Any change to underlying quantities increases the count for
        the appropriate id by 1
    index: A 2D int64 array. The indices for the sparse served model.
    value: A 1D, variable dtype tensor. The parameters
    priority: A 1D tensor, of the same dtype as value. How important a parameter is when freeing
        up memory

    """
    ### Static, inferred properties ###
    @property
    def total(self):
        return self.value.shape[0]
    @property
    def used(self):
        return self.reference.any(dim=-1).sum()
    @property
    def free(self):
        return self.total-self.used

    ### Public functions ###
    def register(self, item: nn.Module):
        """ Register a new memory interface."""

        ## We make an ownership allocation and a version counter, then store it
        #and store the module
        ownership = torch.full([self.total, 1], False, device=self.device)
        version = torch.tensor([0], dtype=torch.int64, device=self.device)

        self.reference = torch.concat([self.reference, ownership], dim=-1)
        self.version = torch.concat([self.latest_version, version], dim=-1)
        self.module_storage.append(item)

    def release(self, id):
        """ Release all memory associated with the reservation """
        self.reference[:, id] = torch.full([self.total], False, device=self.device)
        self.version[id] += torch.tensor(1, device=self.device)
    def reserve(self,
                id: torch.Tensor,
                index: torch.Tensor,
                value: torch.Tensor,
                priority: torch.Tensor):
        """

        Attempts to reserve memory sufficient for the given quantities.
        May end up having to free memory in other places to do so.

        :param id: The id to reserve on
        :param index: The index values to reserve
        :param value: The parameter values to set
        :param priority: The priority weights for the values
        """
    def __init__(self,
                 quantity: int,
                 dtype: torch.dtype,
                 device = None,
                 requires_grad: bool = False,
                 ):

        super().__init__()

        #Setup all seed storage tensors and storage bays

        self.reference = torch.full([quantity, 0], False, device=device)
        self.latest_version = torch.full([0], 0, dtype=torch.int64, device=device)
        self.index = torch.empty([quantity, 2], dtype=torch.int64, device=device)
        self.value = torch.empty([quantity], dtype=dtype, device=device, requires_grad=requires_grad)
        self.priority = torch.ones([quantity], dtype=dtype, device=device)
        self.module_storage = nn.ModuleList()

        #Store buffers and parameters.

        self.register_buffer('reference', self.reference)
        self.register_buffer('latest_version', self.latest_version)
        self.register_buffer('index', self.index)
        self.register_buffer('priority', self.priority)
        self.value = nn.Parameter(self.value, requires_grad=requires_grad)

        self.device = device
