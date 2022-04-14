import torch
import numpy as np
from torch import nn




class ParamPool(nn.Module):
    """
    A large block of memory reserved for parameter purposes.
    Importantly, it can be shared between different layers.
    """

    @property
    def total(self):
        """ How much memory is availible in all"""
        return self._memory.shape[0]

    @property
    def used(self):
        """ How much memory has already been used"""
        return torch.sum(self._allocation[:, 0] != -1)

    @property
    def unused(self):
        """ How much memory is currently available"""
        return torch.sum(self._allocation[:, 0] == -1)
    def __init__(self,
                 quantity: int,
                 capture_quantity: int,
                 dtype=torch.float32,
                 device=None):
        # Start torch
        super().__init__()

        if device is None:
            device = self.device()
        if dtype is None:
            dtype = torch.float32

        # Asserts
        assert isinstance(quantity, int)
        assert isinstance(dtype, torch.dtype)
        assert isinstance(device, torch.device)

        # Setup memory reservation system
        memory = nn.Parameter(torch.empty([0, 1], dtype=dtype, device=device, requires_grad=True))
        self.register_parameter('_memory', memory)




        # register the parameters and buffers.
