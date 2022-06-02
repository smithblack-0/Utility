"""


ANAC Memory

Activity Normalized Auto Calibrating Memory consists
of several mechanisms working in tandem.

- TASK SELECTION

Task Selection consists of a process by which a series of relevant facts are returned
for a particular situation. Let a series of queries E exist. Learnable key matrix K
and value matrix V exist, and have attention performed.

- TASK NORMALIZATION:

The activity for the various memory slots is tracked. More active slots recieve
less urgent updates. Infrequently accessed locations, meanwhile, recieve large updates
when they are found to be useful.

- RUNNING CALIBRATION

Tasks for which the query and key are very similar are considered to be
alike. The most alike task has a proportion of the query inserted into the
similar slot on the key, as a mean across all examples and across all batches.

This means as query shape changes, to some degree the layer can follow along
without being trained to maximize information, simply by asking what the most
beneficial information for the moment would be.

"""
from typing import List
from Utility.Torch.Learnables import Layers

import torch
from torch import nn

class NIIMU(nn.Module):
    """
    Normalized Impressions Internal Memory Unit
    """
    @property
    def burnout(self):
       return 1- torch.softmax(self._Activity, dim=-1)   #(head, mem)

    def rate_hook(self, grads: torch.Tensor):
        return tensor*self.burnout

    def __init__(self,
                 embedding_width: int,
                 memory_width: int,
                 integration_rate: List[float],
                 norm_decay_rate: List[float],
                 ):
        """

        :param embedding_width: the width of the embeddings
        :param memory_width: How large the internel memory should be
        :param integration_rate: The rates of impression integrations for the various heads
        :param norm_decay_rate: The rates of norm decay for the various heads.
        """

        super().__init__()

        #Validate and fetch head width
        assert len(integration_rate) == len(norm_decay_rate)
        head_width = len(integration_rate)


        #Store constants

        self.integration_rate = nn.Parameter(torch.Tensor(integration_rate), requires_grad=False)
        self.decay_rate = nn.Parameter(torch.Tensor(norm_decay_rate), requires_grad=False)


        #Create query, collapse projector.

        self._QueryProj = Layers.Linear(embedding_width, [head_width, embedding_width])
        self._CollapseProj = Layers.Linear([head_width, embedding_width], embedding_width)

        #Create Key, Value memory bank.
        key_parameters = torch.zeros([head_width, memory_width, embedding_width])
        value_parameters = torch.zeros([head_width, memory_width, embedding_width])

        nn.init.kaiming_uniform_(key_parameters)
        nn.init.kaiming_uniform_(value_parameters)

        self._Key = nn.Parameter(key_parameters, requires_grad=True)
        self._Value = nn.Parameter(value_parameters, requires_grad=True)

        #Create Activity Tracker. Used for normalization..

        activity = torch.zeros([head_width, memory_width])
        activity = nn.Parameter(activity, requires_grad=False)
        self._Activity = activity
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """

        :param tensor: A (..., items, embedding) shaped tensor to perform memory
            lookup on.
        :return: A (..., items, embedding) shaped tensor
        """

        #Prep the query and key for attention.
        query = self._QueryProj(tensor) #(...,items, head, embedding)
        query = query.transpose(-2, -3) #(..., head, items, embedding)
        key = self._Key.transpose(-1, -2) #(head, embedding, mem)

        #Perform scoring. Establish update sites. Establish current activity

        score = torch.matmul(query, key) #(..., head, items, mem)

        new_activity = score.transpose(-2, -3).flatten(0, -3).mean(dim=0) #(head, mem)
        decay_rate = self.decay_rate.unsqueeze(-1)
        self._Activity = self._Activity*(decay_rate) + (1-decay_rate)*new_activity

        #Update key with impressions.
        with torch.no_grad():
            decay_rate = self.integration_rate.unsqueeze(-1)
            update_matrix = torch.softmax(score ,dim=-2).transpose(-1, -2) #(head, mem, items)
            key_update = torch.matmul(update_matrix, query).flatten(0,-3).mean(dim=0) #(head, mem, embeddings)
            new_key = self._Key*(1-decay_rate) + decay_rate*key_update
            self._Key.set_(new_key)

        #Attach rate alternations to value backprop

        hook = self.make_rate_hook(burnout)


        self._Value.register_hook()
        key_update = key_update.unsqueeze(-2).unsqueeze(-1) #(head, 1, mem, 1)
        key_update = key_update*query.unsqueeze(-2) #(..., head, items, mem, embedding)


        key_update = self._Key*(1-decay_rate) + burnout*decay_rate*
