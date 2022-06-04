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
from typing import List, Optional
from Utility.Torch.Learnables import Layers

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed.optim import _FunctionalAdam


class PIU(nn.Module):
    """
    Parameter Injection Unit.

    Parameter Memory are large blocks of parameters
    which are compatible wtih an embedded stream
    as though they are embeddings themselves.

    The process of Parameter Injection is a process
    of conditionally injecting whole blocks of parameters,
    into a running embedded stream as though it were an
    embedding itself. Two tasks exist. First, the module
    must figure out what parameter block to inject, and
    when. Second, the module must train the parameter
    blocks to provide useful context.
    """
    def __init__(self, ):

        super().__init__()

        pass


class AutoCalibrationInjector(nn.Module):
    """
    AutoCalibrationInjector

    It is the case that attention is used to perform parameter injection contextualization
    by presenting a trainable key, value parameter bank, and letting the key select the
    value bank to draw from. By this method, multiple sets of unchanging context can be
    injected when the model thinks the input roughly matches a particular pattern.

    The ASPIU module learns passively and autocalibrates. A feedback loop is incorporated between the key
    calibration and key parameter tensors. When a particular query vector is desired more, it modifies
    the key parameters to encourage the selection of such a vector. This in turn causes more of that
    vector to be emphasized in the attention step. This causes the key calibration to start to orient
    in this direction, as further examples are seen. Eventually, it orients appropriately, and the
    dropout penalty causes the key parameters to decay away. This then repeats. This cycle, it must be
    noted, also will automatically deal with minor drift.

    The module adjusts its learning rates to ensure all resources are used. The attention scores
    are tracked in the "activity" parameter, which is a running average of all attention scores.

    This is in turn used to calculate the curiosity. The curiosity is proportional to the inverse
    of the activity, meaning that a frequently accessed parameter bank has a low curiosity.
    The rate of gradient update and calibration update is decreased when curiosity is low.
    Meanwhile, when curiosity is high, it is increased. This ensures when rarely seen circumstances
    come about, we learn a lot from them all at once. Meanwhile, if we have seen it all before,
    we only adjust our parameters a little each time, since we will see many examples.

    """
    @property
    def curiosity(self) -> torch.Tensor:
        """
        Curiosity represents how often a particular memory
        access has been seen. The less often, the higher the curiosity.

        Curiosity works from activity, which is a running average of the
        attention softmax. It is the reciprical of this average, normalized by
        the activity width.
        :return: A tensor
        """
        curiosity = (1-self.activity)
        return curiosity
    def rate_hook(self, grads: torch.Tensor)-> torch.Tensor:
        """
        An infrequent example should have a larger update emphasis. This is the idea
        behind the rate hook. The hook itself is something which is placed on the
        key and value parameters, and encourages larger update gradients on infrequently
        used parameter blocks.

        :param grads: The backprop gradients
        :return: A tensor. The final backprop gradients.
        """
        return grads*self.curiosity.unsqueeze(-1)
    def calibration_trainer(self):
        return self.trainer
    def __init__(self,
                 embedding_width: int,
                 memory_width: int,
                 heads: int,
                 decay_rate: float = 0.999,
                 integration_lr = 0.001,
                 dropout: float = 0.9,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 ):
        """
        The number of heads is indicated by the length of integration_rate and
        norm_decay_rate. Integration rate and norm_decay_rate both indicate,
        for a given batch, what percentage of the original state to keep. This
        being said, it is not actually directly corrolated to this - the curiosity
        also modifies how fast updates occur.

        :param embedding_width: the width of the embeddings
        :param memory_width: How large the internel memory should be
        :param integration_rate: The rates of impression integrations for the various heads
        :param norm_decay_rate: The rates of norm decay for the various heads.
        :param dropout: How strong the feedback dropout is.
        :param grad_rate: How strong the grad curiosity correction is.
        :param curiosity_clamp: The maximum curiosity multiplier.
        """

        super().__init__()

        #Validate and fetch head width
        head_width = heads

        assert embedding_width % head_width == 0
        assert 1 >= decay_rate >= 0
        assert 1 >= dropout >= 0


        head_embeddings = embedding_width//head_width
        #Setup constants and buffers

        embedding_width = torch.tensor(embedding_width, dtype=torch.int32, device=device)
        integration_rate = torch.tensor(integration_lr, dtype=dtype, device=device)
        decay_rate = torch.tensor(decay_rate, dtype=dtype, device=device)
        activity = torch.softmax(torch.zeros([head_width, memory_width], dtype=dtype, device=device), dim=-1).detach()

        self.dropout = dropout
        self.register_buffer('embedding_width', embedding_width)
        self.register_buffer('integration_rate', integration_rate)
        self.register_buffer('decay_rate', decay_rate)
        self.register_buffer('activity', activity)

        #Create Key, Value, Query, Collapse functions and parameters
        key_calibration = torch.zeros([head_width, memory_width, head_embeddings], requires_grad=True)
        key_parameters = torch.zeros([head_width, memory_width, head_embeddings])
        value_parameters = torch.zeros([head_width, memory_width, head_embeddings])

        nn.init.kaiming_normal_(key_parameters)
        nn.init.kaiming_uniform_(value_parameters)

        self.register_buffer('CalibrationBuffer', nn.Parameter(key_calibration, requires_grad=True))
        self.CalibrationBuffer = key_calibration

        self.Query = Layers.Linear(embedding_width, [head_width, head_embeddings])
        self.KeyParameters = nn.Parameter(key_parameters, requires_grad=True)
        self.Value = nn.Parameter(value_parameters, requires_grad=True)
        self.Dehead = Layers.Linear([head_width, head_embeddings], embedding_width)

        #Create regularization, register rate modification hooks, and setup the
        #calibration trainer

        self.KeyParameters.register_hook(self.rate_hook)
        self.CalibrationBuffer.register_hook(self.rate_hook)
        self.Value.register_hook(self.rate_hook)
        self.Dropout = nn.Dropout(dropout)
        self.calibration_loss = nn.L1Loss()

        #Setup Adam parameters for calibration buffer

        self.trainer = torch.optim.Adam([self.CalibrationBuffer], lr=integration_lr)


    def forward(self, tensor: torch.Tensor, total_drop: bool = False) -> torch.Tensor:
        """

        :param tensor: A (..., items, embedding) shaped tensor to perform memory
            lookup on.
        :return: A (..., items, embedding) shaped tensor
        """''


        #Prep the query, key, value for attention.
        query = self.Query(tensor) #(...,items, head, embedding)
        query = query.transpose(-2, -3) #(..., head, items, embedding)
        key = self.Dropout(self.KeyParameters) + self.CalibrationBuffer #(head, mem, embedding)


        value = self.Value #(head, mem, embeddings)

        #Perform scoring, then update the activity, and the calibration. Take into
        #acount how frequently a particular channel has been used by the burnout score. Items
        #with more frequent updates have less effect. Note this automatically updates the gradient
        #rates.
        score = torch.matmul(query, key.transpose(-1, -2))/torch.sqrt(self.embedding_width) #(..., head, items, mem)
        attn_score = torch.softmax(score, dim=-1)

        with torch.no_grad():
            new_activity = attn_score.transpose(-2, -3).flatten(0, -3).mean(dim=0) #(head, mem, items)
            norm_decay_rate = self.decay_rate.unsqueeze(-1) #(head, mem, items)
            self.activity = self.activity*norm_decay_rate + (1-norm_decay_rate)*new_activity


        self.trainer.zero_grad()
        new_calibration_map = score.transpose(-1,-2) #(..., head, mem, items)
        new_calibration_map = torch.softmax(new_calibration_map, dim=-2)
        calibration_vector = torch.matmul(new_calibration_map, query).flatten(0, -4).mean(dim=0) #(head, mem, embedding)
        calibration_loss = self.calibration_loss(calibration_vector, self.CalibrationBuffer)
        calibration_loss.backward(inputs=self.CalibrationBuffer, retain_graph=True)
        self.trainer.step()

        #Finish attention by using score to retrieve values, then collapsing and returning

        attn = torch.matmul(attn_score, value) #(..., head, items, embeddings)
        attn = attn.transpose(-2, -3)
        output = self.Dehead(attn) #(..., items, embedding)
        return output

