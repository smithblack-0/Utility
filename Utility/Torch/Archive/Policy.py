"""

The purpose of this module is to provide infrastructure for the policy
machinery used to dynamically adjust the model during construction for
optimal results.

* each batch contains a number of experiments
* itemized loss used to help calculate penalty
* model size used to help calculate penalty
* Calculated penalty easily fed back to particular cases.

"""

import torch
from torch import nn

class PolicyModule(nn.Module):
    """

    A module for implementing policy logic.

    """

    pass

class PolicyPiece():
    """

    A class representing a particular piece
    of a policy.

    """
    def __init__(self):
        pass

class SlidingInt(PolicyPiece):
    #If policy works...
    pass

class PolicyKernel(nn.Module):
    """

    Takes policy definitions and dynamically
    runs experiments returning differently
    sized kernels.

    """
    pass