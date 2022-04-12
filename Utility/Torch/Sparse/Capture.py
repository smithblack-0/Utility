"""

The Capture module.

This contains classes for capturing information about
gradients on sparse-dense transition points. In particular,

"""
from typing import Callable, Optional

import torch
import torch_sparse
from torch import nn


class StochasticSubSampler(nn.Module):
    """

    A class designed to capture gradient information
    on sparse operations in a reasonably resource efficient
    manner.

    The class performs stochastic subsampling of sample_quantity.
    Note that this means that just because N samples were performed,
    it does not mean those N samples ended up being unique. Still, as
    long as you ensure the result_quantity is above the quantity of
    active tensors,

    --- attributes---

    top_indices: The result_quantity top indices, or the top indices discovered so far, whichever
        is shorter.
    top_values: The result_quantity top values, or the so far discovered top values.
        Whichever is shorter.
    -- methods---

    forward: Performs registered operation
    pop_tk: gets the top k gradients, up to the limit, and releases the
        running average for these entries.
    """
    @property
    def active(self):
        return (self._result_value >= self.fill).sum()
    @property
    def indices(self):
        return self._result_index[:self.active]

    @property
    def values(self):
        return self._result_value[:self.active]
    def pop_tk(self, k):
        """
        pops the top k entries, up to the storage limit.
        These are returned, and the items deactivated.

        :return:
            tensor index of shape (k, 2) and type int64
            tensor value of shape (k) and type float32, representing the gradients so seen.
        """

        if self.active > k:
            k = self.active
        popped_indices, unpopped_indices = self._result_index[:k], self._result_index[k:]
        popped_values, unpopped_values = self._result_value[:k], self._result_value[k:]

        new_indices = torch.empty((k, 2), dtype=torch.int64, device=self.device())
        new_values = torch.full((k,), -1 + self.fill, dtype=torch.float32, device=self.device())

        new_indices = torch.concat([unpopped_indices, new_indices], dim=0)
        new_values = torch.concat([unpopped_values, new_values], dim=0)

        self._result_value = new_values
        self._result_index = new_indices

        return popped_indices, popped_values



    def __init__(self,
                 operator: Callable,
                 result_quantity: int,
                 mask_active: Optional[bool] = False,
                 sample_quantity: Optional[int] = 0,
                 sample_percentage: Optional[float] = 0.2,
                 decay_constant: float = 0.95,
                 device=None,
                 ):
        """

        :param operator: A callable. Will be fed first a SparseTensor, followed by any args and kwargs
        :param result_quantity: The maximum number of results to store in top_indices.
        :param mask_active: Whether to ignore the already active entries when performing updates.
            Be warned, this may slow down updates to a crawl if the tensor gets too dense.
        :param sample_quantity: The number of samples to take from the incoming tensor. Unions itself with
            sample percentage
        :param sample_percentage: The percentage of the sparse tensor shape's area to sample from.
            This is unioned with sample quantity.
        :param decay_constant: the rate at which the running average decays away, if this was dense.
            Will be automatically adjusted to compensate for the current sample quantity.
        """

        super().__init__()

        assert callable(operator)
        assert isinstance(sample_percentage, float)
        assert isinstance(sample_quantity, int)
        assert isinstance(mask_active, bool)
        assert sample_quantity >= 0 and sample_quantity <= 100

        assert isinstance(decay_constant, float)

        # Store constants
        self.operator = operator
        self.sample_size = sample_quantity
        self.sample_percent = sample_percentage

        self.result_size = result_quantity
        self.decay_constant = decay_constant
        self.mask = mask_active

        # Setup result storage. Note that filling a value with a result less than fill will deactivate it.
        self._result_index = torch.full([result_quantity, 2], -1, dtype=torch.int64, device=device)
        self._result_value = torch.full([result_quantity], -1 + self.fill, dtype=torch.float32, device=device)

    def attach_hook(self, index, accumulator, fractional_sparsity):
        """

        :param index:
        :param accumulator:
        :param fractional_sparsity: What percentage of gradients am I sampling
        :return: The accumulator, with the hook attached
        """

        def hook(gradient):
            with torch.no_grad():
                # Decay values and activate the incoming gradients
                adj_decay_constant = 1 - fractional_sparsity * (1 - self.decay_constant)
                decayed_values = adj_decay_constant * self._result_value
                activated_values = gradient.abs()

                # Create combination indices. Then, coalesce the results
                combined_index = torch.concat([index, self._result_index])
                combined_values = torch.concat([decayed_values, activated_values])
                boundaries = combined_index.max(dim=0)
                combined_index, combined_values = torch_sparse.coalesce(combined_index,
                                                                        combined_values,
                                                                        boundaries[0],
                                                                        boundaries[1])

                # Perform an argsort on the combined values. Keep only the
                # top results.

                sort_index = torch.argsort(-combined_values)
                top_sort_index = sort_index[:self.result_size]

                # Get the final index for storage

                final_index = combined_index[top_sort_index]
                final_value = combined_values[top_sort_index]

                # Store it

                self._result_index[:] = final_index[:]
                self._result_value[:] = final_value[:]

        #Attach hook. Return
        accumulator.register_hook(hook)
        return accumulator

    def forward(self,
                sparse: torch_sparse.SparseTensor, *args, **kwargs):
        """

        The sample forward. Will generate hook to catch
        backwards gradients, and store them.

        :param sparse: The tensor to perform sparse operations on
        :param args: Any additional parameters to feed the operator
        :param kwargs: Any additional kwargs to feed the operator
        :return: The result of the operation
        """
        # Calculate sample size and other constants

        device = self.device()
        numel = sparse.size(0) * sparse.size(1)
        sample_size = max(self.sample_size, round(numel * self.sample_percent))

        # Generate unstrided index, remove active if so masked,
        index = torch.randint(0, numel, [sample_size], device=device, dtype=torch.int64)
        if self.mask:
            mask = sparse.storage.row()*sparse.size(0) + sparse.storage.col()
            mask = (index.unsqueeze(0) == mask.unsqueeze(-1)).any(dim=0).logical_not()
            selector = torch.arange(index.shape[0], device=device).masked_select(mask)
            index = index[selector]

        #Create proper index. Find boundaries. Coallese
        row = index.div(sparse.size(0), rounding_mode="floor", dtype=torch.int64) * sparse.size(0)
        col = torch.remainder(index, sparse.size(0))
        index = torch.stack((row, col))
        boundaries = index.max(dim=0)
        index, _ = torch_sparse.coalesce(index, None, boundaries[0], boundaries[1])

        if self.mask:
            #Of the region I am responsible for tracking, figure out what percent is actually
            #being montored.
            fractional_sparsity = index.shape[0] / (numel -sparse.nnz() + 1e-4)
        else:
            fractional_sparsity = index.shape[0]/numel

        # Make the accumulator, and attach its hook
        accumulator = torch.full([index.shape[0]], self.fill, device=device, requires_grad=True)
        accumulator = self.attach_hook(torch.stack([row, col]), accumulator, fractional_sparsity)

        # Make the sampler. This will actually interact with the operator
        sampler = torch_sparse.SparseTensor(row=row, col=col, value=accumulator)

        # Add the sampler and source tensor together. Then perform the operation
        sparse_final = torch_sparse.add(sparse, sampler)
        return self.operator(sparse_final, *args, **kwargs)
