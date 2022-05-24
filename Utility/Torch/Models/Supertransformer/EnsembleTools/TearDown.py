from typing import Optional, Tuple

from torch import nn

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
    """
    #Things I need to know:
    #   What the channels to tear down are
    #   How many catagories to develop per channel.
    #   What loss function to use per channel.
    #   What the names of the labels are in auxilary, per channel.

    def __init__(self, ):
