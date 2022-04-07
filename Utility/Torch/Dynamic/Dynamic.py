#All must be compatible

# DynLength

import torch
from torch import nn
from typing import Sequence, List, Union, Dict
"""

DModule: An extension of torch's Module. Collects DPolicy and DParameter items
DPolicy: A object capable of generating Index objects. Collected by DModule
DTensor: A object which acts much like a torch tensor, but which may have index files for shape parameters.
    Can only interact with DParameters and other DTensor's, but will automatically convert back to a
    torch tensor once all index shapes are gone. 
DParameter: A object which acts as a dynamic parameter. May be initialized with a policy. May be
    sliced by an index. Will automatically initialize needed tensors to match incoming DTensor shape.
    Collected by DModule.

"""

class PopulationManager():
    """

    


    """

class DModule(nn.Module):
    """

    Collects Dynamic Parameters and regular
    parameters

    --- Methods ---

    policies: A generator yielding the policies of this and all subordinate modules
    dparameters: A generator yielding all the dynamic parameters for this and all subordinate modules

    --- Attributes ----

    _policies: The policies on this module
    _dparameters: The dynamic parameters on this module


    _policies: The policies of this particular module

    """
    pass
class DOperatorMixin():
    """


    """


class DTensor(torch.Tensor):
    """

    A Dynamic Tensor.

    Contains, per batch item, information
    indicating which if any dynamic properties are
    active.

    """

    def __init__(self,
                 data: Sequence[torch.Tensor],
                 shapes: Sequence[Sequence[Union[int, torch.Tensor, None]]],
                 *args,

                 policies: Union[Sequence[Dict[int, SubAction]], None] = None,
                 **kwargs):
        
        #Sanity check that items exist for each subcomponent of the batch.
        assert len(data) == len(shapes), "Data did not contain one shape per batch dim"

        #Sanity check that inputs
        for tensor, action, dynamic_shape in zip(data, shapes):

            #Check that any indirect references are in fact

            assert torch.is_tensor(tensor), "Item in sequence data was not tensor"
            data_shape = tensor.shape







        #Start torch
        super().__init__(data, *args, **kwargs)
        
        #Store data, actions, shape


        
        



class DParameter(nn.Parameter):
    pass

class DOptim(torch.optim.Optimizer):
    """
    A base class for optimization in a dynamic environment.

    Dynamic parameters are not dereferenced until update time,
    and algorithms are expected to use DKernels to track parameters
    when dynamic updates may be expected.

    """

