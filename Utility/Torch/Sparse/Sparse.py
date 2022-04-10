import torch
import torch_sparse
import torch_scatter

from torch import nn
from typing import Union, Optional, Sequence


class SparseParameter(nn.Parameter):
    """

    A class to build a sparse compatible pseudodynamic parameter
    manager. One may request a region of memory to be set aside for
    a particular parameter, after which the region can be grown or
    pruned using the associated functions.

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


        assert excess_reservations >= 0

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
                            reservation_shape=shape,
                            requires_grad = requires_grad,
                            )

        #Grow the initial tensor, then return the parameter
        item.grow_(row=row, col=col, value=value)
        return item





    def prune_(self,
               threshold: Optional[float] = None,
               rel_percentage: Optional[float]=None,
               abs_percentage: Optional[float] = None):
        """
        Prunes away excess parameters. These are transparently deactivated
        and kept around for growth cycles.

        Three modes exist. One can discard the parameters whose absolute
        value falls below threshold, the bottom rel_percentage active parameters,
        or even just ensure that abs_percentage total parameters are shut off.

        These modes are all exclusive

        :param threshold: The threshold. Items whose absolute value are below this are pruned
        :param rel_percentage: The percentage. The bottom x percentage of active parameters is pruned.
        :param abs_percentage: The bottom abs_percentage parameters are trimmed, from the total parameters
            If these are already inactive, nothing changes.
        """

        #Sanity check

        if threshold is not None:
            assert threshold >= 0, "Threshold was negative"
            assert rel_percentage is None, "threshold and rel_percentage cannot be active at once"
            assert abs_percentage is None, "threshold and abs_percentage cannot both be active at once"
        if rel_percentage is not None:
            assert 100 >= rel_percentage and rel_percentage >= 0, "Percentage must be between 0 and 100"
            assert threshold is None, "rel_percentage and threshold cannot both be active"
            assert abs_percentage is None, "rel_percentage and abs_percentage cannot both be active"
        if abs_percentage is not None:
            assert 100 >= abs_percentage and abs_percentage >= 0, "Percentage must be between 0 and 100"
            assert threshold is None, "abs_percentage and threshold cannot both be active"
            assert rel_percentage is not None, "rel_percentage and threshold cannot both be active"

        #Get variables
        value = self.sparse.storage.value()
        row = self.sparse.storage.row()
        col = self.sparse.storage.col()

        #Get the number of nonpassing values.
        if rel_percentage is not None:
            num_failed = round(rel_percentage / 100. * value.shape[0])
        elif threshold is not None:
            threshold_result = torch.abs(value) > threshold
            num_failed = threshold_result.numel() - threshold_result.sum()
        elif abs_percentage is not None:
            num_required_deactive = round(abs_percentage/100.*self.total_param_space)
            diff = num_required_deactive-self.total_inactive
            num_failed = max(diff, 0) #Do not go reactivating things.
        else:
            raise RuntimeError("This should not be possible.")


        #Perform an index sort. Strip it apart into the failing and passing sections.
        sorted_results = torch.argsort(torch.abs(value))
        failed_indices, passed_indices = sorted_results[:num_failed], sorted_results[num_failed:]

        #Go and update the parameter index tracker regarding what parameters are active
        #and what ones have failed. Do this by pulling

        active_param_pass = self.active_index[passed_indices]
        active_param_fail = self.active_index[failed_indices]

        self.active_index = active_param_pass
        self.inactive_index = torch.concat([self.inactive_index, active_param_fail], dim=0)

        #Go slice out row, col, value information and update the sparse storage.

        new_rows = row[passed_indices]
        new_cols = col[passed_indices]
        new_values = value[passed_indices]
        self.sparse = torch_sparse.SparseTensor(row=new_rows, col=new_cols, value=new_values)

    def grow_(self,
            row: Optional[torch.Tensor] = None,
            col: Optional[torch.Tensor] = None,
            value: Optional[Union[torch.Tensor, int, float]] = 0,
            discard_unused: Optional[bool] = True):
        """

        A function capable of inserting new connections into new
        parameters.
        :param row:
            A int64 tensor of row indices
        :param col:
            A int64 tensor of column indices
        :param value:
            A value to place at this activated parameter.
        :param discard_unused:
            Whether or not to throw an error when more indices are defined then
            there are spare parameters, or to instead slice out as long a section
            as we can fit and then discard the rest
        """

        #Sanity check, broadcast conversions


        assert isinstance(value, (torch.Tensor, int, float)), "Growth value must be tensor, int, or float"
        if isinstance(value, (int, float)):
            value = torch.full(row.shape[0], value)

        #Handle case where index is too long
        if value.shape[0] > self.inactive_index:
            if discard_unused:
                value = value[self.inactive_index]
                row = row[self.inactive_index]
                col = col[self.inactive_index]
            else:
                raise IndexError("Insufficient parameters to grow for vector of length %s" % row.shape[0])

        #Reactivate needed additional parameters

        length = row.shape[0]
        newly_active, remaining_inactive = self.inactive_index[:length], self.inactive_index[length:]
        self.active_index = torch.concat([self.active_index, newly_active])
        self.inactive_index = remaining_inactive

        #Construct new sparse representation

        if self.sparse is not None:
            old_row = self.sparse.storage.row()
            old_col = self.sparse.storage.col()
            old_val = self.sparse.storage.value()

            row = torch.concat([old_row, row])
            col = torch.concat([old_col, col])
            value = torch.concat([old_val, value])

        self.sparse = torch_sparse.SparseTensor(row=row, col=col, value=value)
    def __init__(
            self,
            reservation: int,
            reservation_shape: Optional[Sequence[int]] = (1,),
            requires_grad: Optional[bool] = True,

    ):
        """
        Initializes the sparse parameter memory block. This is in essence just
        a promise that x amount of memory will be reserved for the construction
        of sparse parameters. .grow_ is needed to make it useful.

        :param reservation: An int. How much distinct memory chunks to reserve
        :param reservation_shape: A sequence of ints, representing the shape of each
            reservation.
        :param requires_grad: Whether a gradient will be needed on the parameter.
        """

        assert isinstance(reservation, int), "reservation must be an int"
        assert reservation >= 0, "Reservation was negative."

        for item in reservation_shape:
            assert isinstance(item, int), "Item in reservation shape was not int"
            assert item > 0, "item in reservation shape was less than 1"

        shape = [reservation, *list(reservation_shape)]
        value = torch.empty(shape)

        super().__init__(value, requires_grad)

        #Start up the memory tracker
        self.sparse = None
        self.active_index = torch.empty([])
        self.inactive_index = torch.arange(reservation)

