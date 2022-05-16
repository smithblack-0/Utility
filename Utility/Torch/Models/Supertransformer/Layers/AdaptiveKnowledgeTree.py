"""

This module centers around a priority-ordered
knowledge distillation centered process known as
Adaptive Knowledge Memory Tree Distillation. It is designed
to allow efficient retrieval of knowledge deeply stored knowledge along with


The purpose of such a method is to allow many, many, many stored
trees of information to exist in parallel, to be able to forget
unused information, and to be able to efficiently lookup needed information.

## KnowledgeTree Distillation

The process consists of taking an incoming TextStream and compacting it down into a sequence
of tensors of decreasing width, the knowledge "tree". The widths of these tensors are statically
determined when creating the model, all incoming tensors are folded into these using attention or
whatever you wish, and each layer is conditioned by the prior layer.

After construction comes utilization. The KnowledgeTree may be queried, as in a transformer.
When this happens, the various TreeLayers are utilized in an additive manner, with the top (smallest)
tree level going off, then the results being added to the next one, then the next one, and so on.

Once distillation is done, it will become part of a broader memory tree. It is the case
that at this point the end tree tensors become learnable.

## Adaptive Setup: Regularization.

For each of the lookups mentioned above, it is
the case that going through the entire tree may take a large amount of time when considered across
many, many, many different examples. Particularly if many memories coexist it may not be desirable to
go through each memory every time.

This is where the adaptive portion kicks in. If we could somehow ensure that more frequently used
information comes first, we could cause the above KnowledgeTree to produce a situation where a
prediction might be made whether the next layer will be useful. Then, we can just go ahead and
only go as far as needed to make a decent prediction.

This is accomplished using two methods. These both operate on the magnitude of the encoding
updates to the query for the above. They are, respectively, Ranking Enforcement and
L1 Regularzation.  Ranking Enforcement uses a differential ranking function, sources from https://github.com/teddykoker/torchsort,
in order to rank the magnitudes of the query updates from each layer. Then,
cross entropy is performed with the rank, demanding the top level layer possess the largest
magnitude, then the second with the second largest, etc. Paried with this, the magnitudes
are also run through a L1 norm. These are tweaked to ensure the ranking enforcement is always stronger
than the L1 Regularization. The L1 regularization naturally encourages the magnitudes to drive to zero
when unneeded, and the ranking means that this always comes in a specified order.

## Adaptive Query:

Finally, we reach the adaptive portion. The adaptive process in a nutshell consists of
going along each KnowledgeTensor in the tree and using it to make an update,
then asking if that update has a magnitude above a particular threshold. Due to the order
enforcement performed above, it should be the case that when the magnitude passes below the threshold,
all subsequent magnitudes for the query will also be below the threshold.

How to train on a new sample, and even how to train originally, does need consideration, however.
In particular, we need to know whether it is the case that something totally new is being thrown at us.
One way to accomplish this is to go about watching whether our configuration is sane as we go along.
To do this, we introduce two mechnisms: If it is the case the ranking is not being properly enforced, a flag is triggered forcing a complete
lookup for that query, and if the magnitude is terminal, we go one step further anyhow and check that
it is still terminal.

Once we determine it is terminal, we remove that query from our lookup. We then proceed onto the next
layer and continue.

## Adaptive Dropping

It may end up being the case that particular bits of information will be of absolutely no
usage to the model. Perhaps somewhat suprisingly, this can actually be detected. During the attention
step of the lookup procedure, it is the case that one will have a matrix indicating for a particular
query how strong the corrolation will be with particular values. This can be utilized in reverse to
see how strongly the values are corrolated with the queries.

We assign each and every bit in the tree besides the top level bits a score which varies
Every time a lookup is performed, it is the case that the score decays exponentially a tiny bit. Meanwhile,
during each lookup step, the queries are summed over and the results added to the subsequent
scores. values which are infrequently queried will naturally begin to decay further and further. Eventually,
they will travel below a particular value, at which point they will just be dropped. As time goes on,
only the most pertinent features will stick around, things that can be used to condition other mistakes.

## MEMORY

A new KnowledgeTree may be inserted into memory at any time. If this happens and sufficient
room does not remain, it is the case that the tree may drop items

##

"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

from Utility.Torch.Learnables import Layers

class MemoryTree():
    """
    A class representing a collection of
    tree tensors. Capable of performing
    informative lookups along with
    updating autonomously.
    """

    def append(self, treeTensors: List[torch.Tensor]):

    def __init__(self, storage_size: int, decay_rate: float):

        self._activitystorage:
        self._treestorage: List[List[torch.Tensor]]

class PriorityTreeTensor():
    """
    A class representing a particular tree. Capable
    of performing lookup and update autonomously,
    and can auto discard as well.
    """
    def __init__(self,
                 treeTensors: List[torch.Tensor],
                 activities: Optional[List[torch.Tensor]] = None,
                 decay_rate: Optional[float] = None,
                 ):

        if activities is None:
            activities = [torch.ones_like(item) for item in treeTensors]
        if decay_rate is None:
            decay_rate = 0.01






class DistillationModel(nn.Module):
    """
    Performs distillation on an entire incoming stream

    Creates a sparse prioritized tree tensor, and
    a sparse activity tensor. Returns them as a
    priority tree.
    """
    pass

class Lookup(nn.Module):
    """
    Looks up information in a particular priority
    """














class KnowledgeEncodeLayer(nn.Module):
    """
    Encodes an individual knowledge layer, given the incoming
    tree_stream, text_stream, and memory. Stores the knowledge
    tensor and interconnections tensor, then returns them
    """
    def __init__(self,
                 d_text: int,
                 d_memory: int,
                 d_tree: int,
                 layer_width: int,
                 ):
    def forward(self,
                knowledge_stream: torch.Tensor,
                text_stream: torch.Tensor,
                memory: torch.Tensor
                ):





class TreeEncodeLayer(nn.Module):
    """

    Description:

    A unit to generate a single new set of tree parameters
    from the incoming tensors. This includes creating
    a new Score_Op and a new Tree Level, then storing them
    in the tensor stream

    """
    @staticmethod
    def _scaled_dot_product_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            dropout_p: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        A modified version of torch's _scaled_dot_product_attention,
        utilizing the sigmoid function rather than the softmax function

        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            dropout_p: dropout probability. If greater than 0.0, dropout is applied.
        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.
            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q, k.transpose(-2, -1))

        attn = F.sigmoid(attn)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn
    def __init__(self,
                 d_stream: int,
                 d_memory: int,
                 d_final: int,
                 width: int,
                 heads: int,
                 dropout: float,
                 dtype: torch.dtype):

        assert d_final % heads == 0
        assert d_stream % heads == 0
        assert d_memory % heads == 0

        super(TreeEncodeLayer, self).__init__()

        #Create norms and memory conditioning

        norm = nn.LayerNorm(d_final)

        #Create seed starting layer and memory conditioning attention
        seed = torch.zeros([width, d_memory],dtype=dtype, requires_grad=True)
        nn.init.kaiming_uniform_(seed)
        seed = nn.Parameter(seed)


        #Create primary attention layers.

        query_projector = Layers.Linear(d_memory, [heads, d_final])
        key_projector = Layers.Linear(d_final, [heads, d_final])
        value_projector = Layers.Linear(d_final, [heads, d_final])
        collapse_projector = Layers.Linear([heads, d_final], d_final)

        #Store entries

        self._dropout = dropout
        self._seed = seed
        self._norm = norm

        self._query_project = query_projector
        self._key_project = key_projector
        self._value_project = value_projector
        self._collapse = collapse_projector

    def forward(self,
                tree_values: torch.Tensor):

        #Setup query, key, value to possess heads and such
        query = self._query_project(self._seed).transpose(-2, -3).contiguous()
        key = self._key_project(tree_values).transpose(-2, -3).contiguous()
        value = self._value_project(tree_values).transpose(-2, -3).contiguous()

        #Perform attention. Return result
        output, score = self._scaled_dot_product_attention(query, key, value, dropout_p=self._dropout)
        return output, score.sum(dim=-1)




class TreeEncode(nn.Module):
    """
    The complete tree encode unit. Applies the encoding process to
    create the complete pseudotree
    """

    @staticmethod
    def make_submodel(
                      d_model: int,
                      widths: List[int],
                      heads: int,
                      dropout: float,
                      dtype: torch.dtype
                      ):
        """
        Makes an instance using the given parameters, and reasonable
        default assumptions.

        :param d_final: The final embeddign dimension
        :param d_memory: The memory embedding dimension
        :param d_stream: The stream embedding dimesion
        :param widths: A list of the widths of the stream heights
        :param heads: The heads to use
        :param dropout: The dropout rate to use
        :param dtype: The dtype
        :return: A new TreeEncodeSubmodel
        """
        layers = []
        for _ in range(total_submodels):
            layer = TreeEncodeLayer(d_model, )
            submodels.append(submodel)
        return TreeEncode(submodels)
    def __init__(self, submodels: List[nn.Module]):

        super(TreeEncode, self).__init__()
        self._submodels = submodels

    def forward(self,
                text_streams: List[torch.Tensor],
                memories: List[torch.Tensor]):

        #Develop reference and tree representation.
        lookup_references: List[List[torch.Tensor]] = []
        tree_representations: List[List[torch.Tensor]] = []

        for submodel, text_stream, memory in zip(self._submodels, text_streams, memories):
            sublookup_reference, subtree_rep = submodel(text_stream, memory)
            lookup_references.append(sublookup_reference)
            tree_representations.append(subtree_rep)

        lookup_references = list(map(list, zip(*lookup_references)))
        tree_representations = list(map(list, zip(*tree_representations)))
        final_lookup_references = [torch.concat(item, dim=-1) for item in lookup_references]



        return lookup_references, tree_representations


class KnowledgeTensor():
    """
    A collector for knowledge from various locations. Stores information
    in a sparse manner. Forgets unused knowledge. Returns it's update
    when a lookup is performed.
    """
    def lookup(self,
               query: torch.Tensor,
               decay: bool = True)-> Tuple[KnowledgeTensor, torch.Tensor, torch.Tensor]:
        """

        :param query:
        :param decay:
        :returns: KnowledgeTensor, query_result, loss
        """

    def append(self,
               lookup_references: List[List[torch.Tensor]],
               access_references: List[List[torch.Tensor]]):



    def __init__(self,
                 layer_index: List[torch.Tensor]
                 layer_access: List[torch.Tensor]
                 layer_descend: List[torch.Tensor]

                 ):

        self.levels = levels
        self.


class Oracle(nn.Module):
    """

    An oracle lookup layer. This knows how to traverse the
    data source it is connected to.

    """

    def __init__(self,
                 regularization_strength: float,
                 d_lookup: int,
                 top_level_width: int):

        super().__init__()

    def forward(self,
                query: torch.Tensor,
                sparse):