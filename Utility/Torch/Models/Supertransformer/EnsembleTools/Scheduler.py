from typing import Optional, Tuple, NamedTuple, List, Callable

import torch
from torch import nn
from torch.nn import functional as F

"""

A scheduler is a device used in loss based teardown. 
It influences which submodels are considered important, and in 
what order. It may respond to the batch number, or details
regarding the logit or regression situation.


"""

class AbstractScheduler(nn.Module):
    """

    This is the abstract scheduler class.

    A scheduler takes in information about the
    current predictive status and batch number, and uses it to adjust
    hyperparameters of the training.

    """
    def __init__(self):
        super().__init__()
    def forward(self, tensor: torch.Tensor, labels: torch.Tensor, batch_num: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward must be implimented")

