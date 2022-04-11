import torch
import torch_sparse
import warnings

from torch import nn
from typing import Union, Optional, Callable


class SubSampler(nn.Module):
    """

    A class designed to capture gradient information
    in a resource efficient manner. It attempts to
    get a reasonable answer to the question of

    "If these tensor entries were actually zero,
    rather than sparse, what would the k largest
    gradients be?"

    It does this by Stochastically sampling from
    the gradient space, storing the largest samples,
    and applying a decay over time.

    It only works well on relatively sparse models.
    """
    @property
    def index(self):
        row = self._backend.sparse.storage.row()
        col = self._backend.sparse.storage.col()
        return torch.stack([row, col], dim=0)

    def required_stochastic_samples(self, L, N):
        
        ### A manually calculated helper funciton. The result is the number of 
        ### stochastic samples it is expected to require to get L distinct quantities out of
        ### N choices, assuming replacement allowed and unique reduced.
     
        p = 1/N #Probability of picking a particular item
        return math.log(1-L*p, 1-p)
        
    def __init__(self,
                operator: Callable,
                max_quantity: int,
                decay_constant: float  = 0.95,
                suppress_live: Optional[bool] = True,
                hysteresis: int = 100,
    ):
        """

        :param operator: A callable. Will be fed first a SparseTensor, followed by any args and kwargs
        :param max_quantity: The maximum number of sparse indices to retain results for.
        :param decay_constant: the rate at which the running average decays away, if this was dense.
        """

        super().__init__()

        assert callable(operator)
        assert isinstance(max_quantity, int)
        assert isinstance(decay_constant, float)
        assert isinstance(suppress_live, bool)

        #Store constants
        self.operator = operator
        self.quantity = max_quantity
        self.decay_rate = decay_constant
        self.suppress = suppress_live

        #Setup result storage.

        self._backend = None

        #Setup optimization
        self._historical = None
        self._hysteresis = hysteresis
    def forward(self, sparse, *args, **kwargs):
        
        
        numel = sparse.size(0)*sparse.size(1)        
        stochastic_samples = self.required_stochastic_samples(self.quantity, numel)
        stochastic_samples = (1+math.sqrt(numel))*stochastic_samples #Padding for variation.
        
        
        if stochastic_samples > numel:
            #It is cheaper in memory, in theory, to run as a pseudo-dense capture
            sampler = self.dense_capture(sparse.size(0), sparse.size(1))
        else:
            #Stochastic capture is the way to go
            sampler = self.sparse_capture(sparse.size(0), sparse.size(1))
            
        def hook(gradients):
            
            
            

        assert isinstance(sparse, torch_sparse.SparseTensor)
        if self.suppress:
            #Suppressed selection will now occur.


        if sparse.sparsity() < 0.5:
            #Not very sparse. Mode will switch to denselike calculation mode,

            row_index = torch.arange(sparse.size(0), device=sparse.device())
            col_index = torch.arange(sparse.size(1), device=sparse.device())
            true_index = torch.meshgrid([row_index, col_index])
            values = torch.zeros([sparse.size(0), sparse.size(1)], device=sparse.device())
            sampler =








        #Get the excluded values if relavent
        if self.suppress:


            row = sparse.storage.row()
            col = sparse.storage.col()
            src_index = torch.stack([row, col])







class SparseParameter(nn.Module):
    """

    A class to build a torch-sparse compatible pseudodynamic parameter
    manager. One may request a region of memory to be set aside for
    sparse parameter construction purposes, after which the region can be grown or
    pruned using the associated functions and reshaped to any size, contingent
    on not exceeding the memory footprint.

    This class is savable

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

    def prune_(self,
               threshold: Optional[float] = None,
               rel_percentage: Optional[float]=None,
               abs_percentage: Optional[float] = None,
               sort: Optional[Union[str, Callable]] = None):
        """
        Prunes away excess parameters. These are transparently deactivated
        and kept around for growth cycles.

        Three modes exist. One can discard the parameters whose absolute
        value falls below threshold, the bottom rel_percentage active parameters,
        or even just ensure that abs_percentage total parameters are shut off.

        These modes are all exclusive. The final parameter is sort, which governs how
        the values are sorted when making the prune decision.



        :param threshold: The threshold. Items whose absolute value are below this are pruned
        :param rel_percentage: The percentage. The bottom x percentage of active parameters is pruned.
        :param abs_percentage: The bottom abs_percentage parameters are trimmed, from the total parameters
            If these are already inactive, nothing changes.
        :param sort: The sorting method used. May be one of a predefined list, or a custom function
            Options are:
                "AscendingAbs (default)": Items have their absolute value taken, then are sorted in
                    ascending order
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
            num_required_inactive = round(abs_percentage/100.*self.total_param_space)
            diff = num_required_inactive-self.total_inactive
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
            row: torch.Tensor = None,
            col: torch.Tensor = None,
            value: Union[torch.Tensor, int, float] = 0,
            discard_unused: Optional[bool] = True):
        """

        A function capable of inserting new connections into new
        parameters.

        :param row:
            A int64 tensor of row indices. 1d, or empty 0d
        :param col:
            A int64 tensor of column indices
        :param value:
            A value to place at this activated parameter.
        :param discard_unused:
            Whether or not to throw an error when more indices are defined then
            there are spare parameters, or to instead slice out as long a section
            as we can fit and then discard the rest
        :return: True or False. True if completed without complication. False if values were
            discarded

        :raises:
            AssertionError, if something is wrong.
            RuntimeWarning, if discard is executed.


        """
        with torch.no_grad():
            #Sanity checks
            assert torch.is_tensor(row), "row was not tensor"
            assert row.dim() == 1, "row was not 1d"

            assert torch.is_tensor(col), "col was not tensor"
            assert col.dim() == 1, "col was not 1d"

            assert row.shape[0] == col.shape[0], "row and col must have the same length"


            sparse_length = row.shape[0]
            assert isinstance(value, (torch.Tensor, int, float)), "value must be tensor, int, or float"
            if isinstance(value, (int, float)):
                value = torch.full([sparse_length], value)
            if value.dim() == 0:
                value = torch.full([sparse_length], value.item())

            assert value.dim() == 1, "value must be 1d"
            assert value.shape[0] == sparse_length, "row and value lengths do not match"

            #Handle case where index is too long
            if sparse_length > self.total_inactive:
                if discard_unused:
                    if self.suppressing_grow_warning is False:
                        message ="""
                                Provided index and values are longer than remaining parameters
                                This may cause unexpected behavior
                                
                                This message will now self suppress.
                                """
                        
                        warnings.warn(message, RuntimeWarning)
                        self.suppressing_grow_warning = True
                    value = value[:self.total_inactive]
                    row = row[:self.total_inactive]
                    col = col[:self.total_inactive]
                    sparse_length = self.total_inactive
                else:
                    raise IndexError("Insufficient parameters to grow for vector of length %s" % sparse_length)

            #Reactivate needed additional parameters

            length = sparse_length
            newly_active, remaining_inactive = self.inactive_index[:length], self.inactive_index[length:]
            self.active_index = torch.concat([self.active_index, newly_active])
            self.inactive_index = remaining_inactive

            self.backend[newly_active] = value #setup parameter
            value = self.backend[newly_active] #get parameterized version


            #Construct new sparse representation. Update

            if self.sparse is not None:
                old_row = self.sparse.storage.row()
                old_col = self.sparse.storage.col()
                old_val = self.sparse.storage.value()

                row = torch.concat([old_row, row])
                col = torch.concat([old_col, col])
                value = torch.concat([old_val, value])

                print(row)

            self.sparse = torch_sparse.SparseTensor(row=row, col=col, value=value)

            #Finish by returning true if entirely successful, or false if content was trimmed.
            if sparse_length > self.total_inactive:
                return False
            return True
    def capture(self,
                callback: Callable,
                quantity: Optional[int] = None,
                rel_percentage: Optional[float] = None,
                abs_percentage: Optional[float] = None,
                threshold: Optional[float] = None,
                include_dead: Optional[bool] = True,
                include_live: Optional[bool] = False,
                ):
        """

        A capture driver class. If called, based on the parameters provided
        the next execution will capture the sparse backpropogation gradients to
        enable execution of algorithms like RigL. In particular, on the next
        monitored backwards call, the N largest gradients are computed, their
        indices found, and the results are then shoved into callback.

        Multiple modes may be executed at once. These will operate in parallel, but
        will tend to find the same results. Fine tuning may be useful.

        :param callback: The callable to be called when the capture is made.

        :param quantity: The quantity of indices to capture. Unioned with rel, abs percentage,
            threshold
        :param rel_percentage: How many captures to make with respect to the active parameters.
            Unioned with respect to quantity, abs_percentage, threshold
        :param abs_percentage: How many captures to make with respect to the total parameters.
            Unioned with respect to quantity, rel_percentage, threshold
        :param threshold: Return all items with gradients above a threshold.
            Unioned with respect to quantity, rel_percentage, abs_percentage.

        :param include_dead:
            Whether to include connections which would not be propogated back to the sparse case.
            Unioned with include_live
        :param include_live:
            Whether to include connections which would be propogated back in the sparse case.
            Unioned with include_dead
        """

        has_set_capture_flag = False
        capture_quantity = 0
        capture_threshold = None
        if isinstance(quantity, int):
            assert quantity >= 0
            capture_quantity = quantity
            has_set_capture_flag = True

        if isinstance(rel_percentage, float):
            assert rel_percentage >= 0 and rel_percentage <= 100
            quantity = round(self.active_index * rel_percentage / 100)
            capture_quantity = max(self._capture_quantity, quantity)
            has_set_capture_flag = True
        if isinstance(abs_percentage, float):
            assert abs_percentage >= 0 and abs_percentage <= 100
            quantity = round(self.total_param_space * abs_percentage / 100)
            capture_quantity = max(self._capture_quantity, quantity)
            has_set_capture_flag = True

        if isinstance(threshold, int):
            assert threshold >= 0
            threshold = threshold
            has_set_capture_flag = True

        if has_set_capture_flag is False:
            raise ValueError("No capture option was chosen")





    def __init__(
            self,
            reservation: int,
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



        self.sparse = None
        self.backend = nn.Parameter(torch.empty([reservation]), requires_grad=requires_grad)

        #Start up the parameter memory tracker
        self.active_index = torch.empty([0],)
        self.inactive_index = torch.arange(reservation)

        #Start flags and event slogs
        self.suppressing_grow_warning = False
        self._capture_quantity = None
        self._capture_dead = False
        self._capture_live = False
        self._threshold = None
        self._callback = None

    def matmul(self, other: torch.Tensor):
        """

        Performs a capture monitored matrix multiply. When a capture is prepped,


        :param other: A dense tensor to multiply
        :return: A dense tensor. The result of matrix mulplication
        """
        assert torch.is_tensor(other), "Second element must be dense tensor"
