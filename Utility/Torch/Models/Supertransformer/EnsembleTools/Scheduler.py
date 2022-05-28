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
    def forward(self,
                tensor: torch.Tensor,
                labels: torch.Tensor,
                batch_num: int,
                epoch: int,
                cum_batch_num: int,
                ensemble_channel: int) -> torch.Tensor:
        raise NotImplementedError("Forward must be implimented")

### General schedulers ###
class BasicEngagement(AbstractScheduler):
    """
    This is a basic engagement scheduler. It simply turns on the submodels in sequence depending
    on the batch number over a period of batch numbers.
    """
    def __init__(self, channel_engagement_factor=200, engagement_rate = 20):
        """
        :param channel_engagement_factor: How many batches apart channels turn on
        :param engagement_rate: The batch width it takes to turn from
            %10 on to %90 on.
        """
        super().__init__()
        self.channel_factor = channel_engagement_factor
        self.rate = engagement_rate
    def forward(self,
                tensor: torch.Tensor,
                labels: torch.Tensor,
                batch_num: torch.Tensor,
                epoch: torch.Tensor,
                cum_batch_num: torch.Tensor,
                ensemble_channel: torch.Tensor) -> torch.Tensor:
        # This should simply wait until
        half_on = ensemble_channel*self.channel_factor
        activation = torch.sigmoid(2*half_on/self.rate)
        return activation

class StaticScheduler(AbstractScheduler):
    """
    Returns the same value, no matter what
    """
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.value = value
    def forward(self,
                tensor: torch.Tensor,
                labels: torch.Tensor,
                batch_num: int,
                epoch: int,
                cum_batch_num: int,
                ensemble_channel: int) -> torch.Tensor:
        return self.value




