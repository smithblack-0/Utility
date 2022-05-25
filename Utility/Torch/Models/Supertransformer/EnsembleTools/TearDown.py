from typing import Optional, Tuple, NamedTuple, List, Callable

import torch
from torch import nn
from torch.nn import functional as F

from Utility.Torch.Models.Supertransformer import StreamTools
from Utility.Torch.Models.Supertransformer.StreamTools import StreamTensor


class AbstractEnsembleTeardown(nn.Module):
    """
    A class responsible for finishing up with
    an ensemble of some sort and narrowing the
    ensemble back down to a single output stream. This is a single
    subinstance.

    -- forward params --

    ensemble_stream: The item from the currently evaluated ensemble.
    cumulative_stream: The item from the last evaluated ensemble. Optional.
    auxiliary_stream: Any additional information the user may want, such as losses
        or training commands. Optional.
    """
    def __init__(self):
        super().__init__()
    def forward(self,
                ensemble_stream: StreamTools.StreamTensor,
                cumulative_stream: Optional[StreamTools.StreamTensor] = None,
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None) \
            -> Tuple[StreamTensor, StreamTensor]:
        """

        :param ensemble_stream: The current stream from the current instance
        :param cumulative_stream: The cumulative stream. From the previous iteration.
        :param auxiliary_stream: Anything from the auxiliary stream
        :return: Two items. First, the cumulative stream tensor. Second, the task streamtensor.
        """
        raise NotImplementedError("Must impliment forward function in EnsembleTeardown")


class CategoricalBoostedTeardown(AbstractEnsembleTeardown):
    """
    A teardown designed to work with categorical data, providing the losses and predictions
    as an output. Performs boosting - each step updates the previous guess, as in XGBOOST,
    and the result is accumulated along the cumulative channel. The incoming ensemble
    stream is routed to the channel output.

    The last dimension will have a logit formed on it.
    """
    #Things I need to know:
    #   What the channel to tear down is
    #   What the catagories are for the channel
    #   What loss function to use.
    #   What the names of the labels are in auxilary, per channel.
    def __init__(self,
                 input_width: int,
                 logit_width: int,
                 channel_name: str,
                 loss_function: Callable,
                 activation_function: Callable,
                 label_names: str):
        super().__init__()

        self.default_width = input_width
        self.logit_width = logit_width

        self.logits = nn.Linear(input_width, logit_width)
        self.channel_name = channel_name
        self.loss_function = torch.jit.script(loss_function)
        self.activation = torch.jit.script(activation_function)
        self.labels = label_names
    def forward(self,
                ensemble_stream: StreamTools.StreamTensor,
                cumulative_stream: Optional[StreamTools.StreamTensor] = None,
                auxiliary_stream: Optional[StreamTools.StreamTensor] = None) -> Tuple[StreamTensor, StreamTensor]:

        tensor = ensemble_stream.isolate([self.channel_name])[0]
        logits = self.logits(tensor)

        if cumulative_stream is not None:
            cumulative_tensor, weights = cumulative_stream.isolate([self.channel_name + 'cumulative', self.channel_name + 'weights'])[0]
            logits = logits + cumulative_tensor
        else:
            weights = torch.ones(self.logit_width)

        activations = self.activation(logits)
        (labels) = auxiliary_stream.isolate([self.labels])
        loss = self.loss_function(activations, logits, reduction="none", weight=weights)
        weights = torch.softmax(loss, dim=-1)

        stream_items = {self.channel_name + 'cumulative' : logits, self.channel_name + 'weights' : weights}
        null_stream = cumulative_stream.keeponly([])
        update_stream = StreamTensor(stream_items, {self.channel_name + 'loss' : loss} )
        merger = StreamTools.StreamMerger([null_stream, update_stream])
        merger.stream.sum()
        merger.losses.sum()
        cumulative_stream = merger.build()

        return cumulative_stream, ensemble_stream








