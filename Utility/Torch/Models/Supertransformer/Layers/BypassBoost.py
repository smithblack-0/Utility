from typing import List, Union

import torch
from torch import nn

class ModelFactory():
    """
    Creates copies of a model on demand.
    One may use it as a @ item
    """

    def __init__(self, model, *args, **kwargs):
        self._kwargs = kwargs
        self._args = args
        self._model = model
    def __call__(self):
        return self._model(*self._args, **self._kwargs)










class ResidualBypassSubmodel(nn.Module):
    """
    A single stream submodel. Processes the
    current input using the sublayer stack.
    """

    def __init__(self,
                interchange_layers: List[nn.Module]
                 ):
        super().__init__()
        layers = [torch.jit.script(layer) for layer in interchange_layers]
        self._layers = nn.ModuleList(layers)
    def forward(self,
                tensor: torch.Tensor,
                memory: torch.Tensor,
                residuals: List[torch.Tensor],
                args: List[torch.Tensor]):


        new_residuals = []
        tensor = tensor
        for i, layer in enumerate(self._layers):
            if len(residuals) == 0:
                residual = torch.zeros_like(tensor)
            else:
                residual = residuals[i]
            tensor = tensor + residual
            tensor, memory = layer(tensor, memory, args)
            new_residuals.append(tensor)
        return tensor, memory, new_residuals




class ResidualBypassModel(nn.Module):
    """
    Given a sequential model, construct a
    residual bypass model of it with the indicated
    number of parallel channels. Returns the
    entire sequence as an output.
    """
    class SubModel(nn.Module):
        """

        A helper class. This represents a single submodel,
        in a bypass compatible format. It accepts a list
        consisting of the input tensors, and prior residuals.

        The inputs must be the residuals and inputs in a list.
        The residuals must be

        """
        def prime(self, channels: List[Union[List[torch.Tensor], bool]]):
            """

            Develop a priming


            :param channels:
            :return:
            """

        def __init__(self, model: nn.Sequential):
            super().__init__()
            self._layers = nn.ModuleList(model.children())

        def forward(self,
                    inputs: List[torch.Tensor],
                    residuals: List[List[torch.Tensor]],
                    maskings: torch.Tensor
                    ):
            tensors = inputs
            output_residuals = []
            for i, layer in enumerate(self._layers):
                residual_count = 0
                subresiduals = residuals[i]
                submask = maskings[i]
                composite_tensors = []
                for j, mask in submask:
                    if mask == True:
                        composite_tensor = tensor[j] + subresiduals[residual_count]
                    else


                composite_tensors = []
                subresiduals = residuals[i]
                for residual, tensor in zip(subresiduals, tensors):
                    if isinstance(residual, torch.Tensor):
                        composite_tensor = tensor + residual
                    else:
                        composite_tensor = tensor
                    composite_tensors.append(composite_tensor)
                tensors = layer(*composite_tensors)
                output_residuals.append(tensors)
            return tensors, output_residuals



    def __init__(self,
                 submodels: int,
                 model_factory: ModelFactory):
        """

        :param submodels: The number of submodels to make
        :param bypass: The bypass directives. For each
            layer in sequential, and then each input into layer,
            indicate whether or not to bypass the input
        :param model_factory:
            The factory model to get information from.
            This should return a sequential.

            Each input argument will be residually bypassed, no questions asked
            This means the output of one feeds into the input of the next.
        """

        super().__init__()
        assert isinstance(submodels, int)
        assert isinstance(model_factory, ModelFactory)

        submodels =
        for




