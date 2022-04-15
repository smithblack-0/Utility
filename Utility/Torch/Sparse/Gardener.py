import torch
from torch import nn
from Parameter import SparseParameter



class Gardener():
    """
    A gardener is capable of managing a large collection of
    SparseParameters. It should be fed a root module, and
    then will provide options for manipulating the parameters.


    """
    def __init__(self,
                 model: nn.Module
                 ):
        #Find and collect all the sparse parameters in the model.
        SparseParams = []
        for item in model.modules():
            if isinstance(item, SparseParameter):
                SparseParams.append(item)
        #Find and collect all the nonsparse parameters in the model
        Params = list(model.parameters())

        #Store
        self._sparameters = SparseParams
        self._parameters = Params



