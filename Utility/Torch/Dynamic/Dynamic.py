#All must be compatible

# DynLength

import torch
from torch import nn

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
class DOperators():
    """

    A holder for a class which makes dynamic tensors and processes them

    """
    @staticmethdo

class DTensor():
    """

    A Dynamic Tensor. Created by interactions between other dynamic tensors
    , dynamic parameters, or tensors and dynamic parameters.

    """
    ## Tensor interface: Operations

    ## Tensor interface: Conditions ##
    def equal(self, other):
        """ Check if the two DTensor's are the same"""
        pass
    def less_equal(self, other):
        """ Check if items in self is less than or equal to other"""
        pass
    def greater_equal(self, other):
        """ Check if items in self is greater than or equal to other"""
        pass
    def not_equal(self, other):
        """ Check if this item and other are not the same."""
        pass
    def any(self):
        """ Return true if any subentries are true"""
        pass
    def all(self):
        """ Return true if all subentries are true"""



    #Dunder methods: Comparison
    def __eq__(self, other):
        return self.equal(other)
    def __le__(self, other):
        return self.less_equal(other)
    def __ge__(self, other):
        return self.greater_equal(other)
    def __ne__(self, other):
        return self.not_equal(other)


class DParameter(nn.Parameter):

class DOptim(torch.optim.Optimizer):
    """
    A base class for optimization in a dynamic environment.

    Dynamic parameters are not dereferenced until update time,
    and algorithms are expected to use DKernels to track parameters
    when dynamic updates may be expected.

    """

