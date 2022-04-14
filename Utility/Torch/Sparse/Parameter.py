import collections
import math
import uuid

import torch
import torch_sparse
import warnings

from torch import nn
from typing import Union, Optional, Callable, Sequence




class SparseParameter(nn.Module):
    """

    A class to build a torch-sparse compatible pseudodynamic parameter
    manager. One may request a region of memory to be set aside for
    sparse parameter construction purposes, after which the region can be grown or
    pruned using the associated functions and reshaped to any size, contingent
    on not exceeding the memory footprint.

    """

    @property
    def total_active(self):
        return self.active_index.shape[0]
    @property
    def total_inactive(self):
        return self.inactive_index.shape[0]
    @property
    def total_param_space(self):
        return self.inactive_index + self.active_index

    @classmethod
    def from_sparse(cls,
                    sparse_tensor: torch_sparse.SparseTensor,
                    excess_reservations: Optional[int] = 0,
                    requires_grad: Optional[bool] = True,
                    ):
        """
        Create a parameter space from an initial sparse tensor.

        :param sparse_tensor: The initial tensor to build with
        :param excess_reservations: The number of extra parameters to reserve space for
        :param requires_grad: Whether a gradient is required or not
        :return:
        """


        assert excess_reservations >= 0, "excess reservation cannot be negative"

        #Get sparse information
        row = sparse_tensor.storage.row()
        col = sparse_tensor.storage.col()
        value = sparse_tensor.storage.value()

        #Get initiation params
        length = row.shape[0] + excess_reservations
        shape = value[0].shape

        #Reserve the required quantity of memory as a parameter.
        item = cls.__init__(length,
                            reservation = length,
                            requires_grad = requires_grad,
                            )

        #Grow the initial tensor, then return the parameter
        item.grow_(row=row, col=col, value=value)
        return item
    def from_tensor(self,
                    tensor: torch.Tensor,
                    mask: torch.Tensor,
                    max_sparsity: float):
    @property
    def status(self):
        """
        Get the status
        :return: index, value
        """
        index = self.index
        value = self.value[index.shape[0]]
        return index, value
    @status.setter
    def status(self, item):
        #Unwravel
        index, value = item

        #Check sane
        assert torch.is_tensor(index)
        assert torch.is_tensor(value)
        assert index.dtype == torch.int64
        assert value.dtype == self.value.dtype
        assert index.dim() == 2
        assert value.dim() == 1
        assert index.shape[0] == value.shape[0]

        #Store
        with torch.no_grad():
            length = index.shape[0]
            self.value[:length] = value
            self.index = index
            self._build()

    def _build(self):
        """
        Rebuilds the sparse tensor based on the stored
        index and values

        """

        index = self.index
        value = self.value[:index.shape[0]]
        self.sparse= torch_sparse.SparseTensor(
                                                row=index[0],
                                                col=index[1],
                                                value=value)
    def __init__(
            self,
            reservation: int,
            dtype: torch.dtype = None,
            device: torch.device = None,
            requires_grad: Optional[bool] = True,
    ):
        """
        Initializes the sparse parameter memory block. This is in essence just
        a promise that x amount of memory will be reserved for the construction
        of sparse parameters. .grow_ is needed to make it useful.

        Also, sets up the capture logic if we will be capturing gradients. This will


        :param reservation: An int. How many distinct parameters chunks to reserve
        :param requires_grad: Whether a gradient will be needed on the parameter.
        :param capture_enabled: Whether a capture buffer will be built, and captures allowed
        :param capture_buffer_percent: What percentage of the reservation size the capture
            buffer should be.

            A capture larger than this cannot be made.
        """

        super().__init__()

        assert isinstance(reservation, int), "reservation must be an int"
        assert reservation >= 0, "Reservation was negative."
        assert isinstance(requires_grad, bool), ("requires grad must be bool", requires_grad)



        self.value = nn.Parameter(torch.empty([reservation],
                                                dtype=dtype,
                                                device=device),
                                    requires_grad=requires_grad,
                                    )
        self.index = torch.empty([0, 2],dtype=torch.int64, device=device)
        self._build()

