"""

This module contains classes and functions which do not themselves contain
trainable parameters but instead can be used to prepare data for further processing.

"""
import math
from typing import List, Union, Optional, Callable

import torch
from torch import nn
import numpy as np
from functools import wraps

from Utility.Torch import Glimpses



class CompositeComponentConverter(nn.Module):
    """

    Composite Component format Converter

    This class is designed to allow the augmentation of the local/banded transformer architecture
    to allow it to perform meaningful and efficient data interchange between both highly local and
    highly global contexts.

    -- fields --

    d_composite: The expected d_model dim of the composite tensor
    d_models: The expected embedding width of the component tensors
    ratios: The ratios components are expected to be in
    lcm: An input composite tensor must be divisible by this.

    -- methods---

    composite: Places the incoming tensor list into composite local format
    component: Breaks the incoming tensor into superlocal component format

    -- details --

    The Local format is gained by zipping. In this format, items are repeated according to ratios
    in various embedding dimensions of the tensor. For instance, if items are given in a 10:1 ratio
    with d_model 2, 2

    --- examples ---

    Lets see all of this in action. Given the following setup:

        tensor = torch.randn([10, 100, 50])
        d_models = [10, 30, 10]
        ratios = [10, 1, 0]
        instance = Architecture.CompositeComponentConverter(d_models, ratios)
        test = instance.component(tensor)

   Test will end up with a list three tensors, for the three different ratios. The
   first one will be shaped as [10, 100,10], the second as [10, 10, 30], and the
   third as [10, 10, 1]. Performing

        test = instance.composite(test)

    Will then cause test to become a shape of the original tensor, [10, 100, 50]

    """
    @staticmethod
    def _calc_lcm(items: List[int]) -> int:
        lcm = torch.tensor(1, dtype=torch.int64)
        for item in items:
            lcm = lcm*item//math.gcd(lcm, item)
        return lcm

    @property
    def _nonzero_ratios(self):
        nonzero = self.ratios > 0
        nonzero = self.ratios.masked_select(nonzero)
        return nonzero
    @property
    def d_total(self):
        return sum(self.d_models)
    @property
    def lcm(self) -> int:
        items: List[int] = self._nonzero_ratios.tolist()
        return self._calc_lcm(items)
    @property
    def inverses(self):
        output: List[int] = []
        for item in self.ratios:
            if item > 0:
                output.append(self.lcm//item)
            else:
                output.append(0)
        return output
    def __init__(self,
                 d_models: List[int],
                 ratios: List[int],
                 reducers: Optional[List[nn.Module]] = None):

        """

        This is initialized with a list of the component d_models, the component
        item ratios, and optionally the component reducers.

        The reducers, if not specified, will be the max function. If they are specified,
        they must be a callable that accepts an arbitrarily shaped tensor and completely
        eliminates the last dimension.

        One special note is that setting a ratio to zero defines the entity as a global item
        Global items are always fully reduced, that is reduced until the number of embeddings
        are one.

        :param d_models: The componentwise embedding dimensions of the model. How much
            is assigned to each particular component. Must be a list of ints. Must all be
            >= 0.
        :param ratios:
            The ratios of the component embedding lengths to one another. [10, 1] for instance
            would be that after component breakdown for every 10 of a there is only 1 of b. Letting
            a ratio be zero will fully reduce it, as a global embedding dimension. Must be a list
            of ints, of length of d_models, and all >= 0
        :param reducers:
            Optional. A list of length d_models, containing callables. These callables
            take in a tensor of arbitrary shape, and reduce the last dimension to nothing.
        """
        super().__init__()

        if reducers is None:
            class max_reducer(nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, tensor):
                    return tensor.max(dim=-1).values
            reducers = [max_reducer() for _ in d_models]
        torch.jit.annotate(List[nn.Module], reducers)

        assert len(d_models) == len(ratios)
        assert len(d_models) == len(reducers)
        for index in range(len(d_models)):
            assert callable(reducers[index])
            assert d_models[index] >= 0
            assert ratios[index] >= 0

        self.d_models = torch.tensor(d_models, dtype=torch.int64)
        self.ratios = torch.tensor(ratios, dtype=torch.int64)
        self.reducers = nn.ModuleList(reducers)

    def composite(self, tensors: List[torch.Tensor])->torch.Tensor:
        """

        this is designed to take a sequence of tensors in superlocal
        component format and place it into composite format. This then
        means that a single value may appear in many places in nearby
        embedding items.

        :param tensors: A list of superlocal tensors. Must have words in the
        appropriate ratios, and the appropriate dimensions
        :return: The local tensors. Superlocal tensors will end up being repeated
            in multiple places if so relevant.
        :raises:
            AssertError: If items are not given in the appropriate ratios, or
            the d_models are not correct.
        """



        d_widths = [item.shape[-1] for item in tensors]
        lengths = [item.shape[-2] for item in tensors]
        final_length: int = self._calc_lcm(lengths)

        assert sum(d_widths) == self.d_total
        assert final_length % self.lcm == 0

        output = []
        repetitions = [item if item > 0 else final_length for item in self.inverses]
        for item, repetition in zip(tensors, repetitions):
            item = torch.repeat_interleave(item, repetition, dim=-2)
            assert item.shape[-2] == final_length
            output.append(item)
        return torch.concat(output, dim=-1)

    def component(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """

        breaks a composite tensor apart into component format.
        In component format, each component has been reduced in length
        by the appropriate ratio, and unpeeled according to the appropriate
        dimension.

        The input requirements are that the number of embeddings is of length
        lcm, and that the embedding width is equal to d_total.

        :param tensor: The tensor to break into components
        :return: A list of components.
        """

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() >= 2
        assert self.d_total == tensor.shape[-1]
        assert tensor.shape[-2] % self.lcm == 0

        output = []
        length = tensor.shape[-2]
        splits = [length//item if item > 0 else 1 for item in self.inverses]
        sections: List[int] = self.d_models.tolist()
        subsections = tensor.split(sections, dim=-1)
        for i, reducer in enumerate(self.reducers):
            item = subsections[i]
            split = splits[i]

            item = item.split(split, dim=-2)
            item = torch.stack(item, dim=-1)
            item = reducer(item)

            output.append(item)

        #Return the result
        return output



