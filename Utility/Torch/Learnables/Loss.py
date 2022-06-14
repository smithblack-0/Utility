import torch
from torch import nn
from torch.nn import functional as F

class CrossEntropyEnsembleBoost(nn.Module):
    """
    Cross entropy with a bit of a twist

    Beginning from the lowest item in the
    ensemble channel, we calculate the
    cross entropy loss, use this as weights
    on the next calculation, then proceed
    to the next channel and do it again.

    The final result will be the sum of
    all the intermediate losses.

    """
    def __init__(self,
                 ensemble_channel: int=-3,
                 label_smoothing: float = 0.0,
                 boost_smoothing: float = 0.2
                 ):
        super().__init__()
        self.channel = ensemble_channel
        self.smoothing = label_smoothing
        self.boost = boost_smoothing

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor):

        # Prep and smooth labels.
        if labels.dtype == torch.int32 or labels.dtype == torch.int64 or labels.dtype == torch.int16:
            labels = F.one_hot(labels, self.logit_width).type(torch.float32)
        labels = (1 - self.smoothing) * labels + self.smoothing / self.logit_width

        #Unbind the ensemble channel, then prep loss and weights for accumulatin
        logits = predictions.unbind(self.channel)
        weights = torch.ones_like(predictions[0])
        loss = torch.tensor([0.0], dtype=predictions.dtype)
        for logit in logits:
            entropy = - labels*torch.log_softmax(logit, dim=-1)
            weighted_entropy = weights*entropy
            loss = loss + weighted_entropy.sum()/logit.shape[-1]
            weights = weighted_entropy*(1-self.boost) + self.boost
        return loss

class CrossEntropyAdditiveBoost(nn.Module):
    """
    Cross entropy with a bit of a twist.

    Each ensemble channel is independently
    processed, and then starting from channel one
    the results are merged. In particular, each
    channel contributes an additive factor to the
    final logits; additionally, each intermediate
    logit is evaluated and the bit losses are used
    to update the cross entropy weights.

    The net result is that if there was a lot of
    loss on the prior layer, this layer will aggressively train to
    minimize further loss.

    """
    def __init__(self,
                 logit_width: int,
                 ensemble_channel: int = 1,
                 label_smoothing: float = 0.0,
                 boost_smoothing: float = 0.3):

        super().__init__()
        self.logit_width = logit_width
        self.channel = ensemble_channel
        self.smoothing = label_smoothing
        self.boost = boost_smoothing
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, data: torch.Tensor, labels: torch.Tensor):

        #Prep and smooth labels.
        if labels.dtype == torch.int32 or labels.dtype == torch.int64 or labels.dtype == torch.int16:
            labels = F.one_hot(labels, self.logit_width).type(torch.float32)
        labels = (1-self.smoothing)*labels + self.smoothing/self.logit_width

        #Prep various channels
        channels = data.unbind(dim=self.channel)
        weights = torch.ones_like(channels[0])
        logits = torch.zeros_like(channels[0])
        loss = torch.tensor([0.0], dtype=data.dtype)

        #Run
        for channel in channels:

            #Update loss
            logits = logits + channel
            entropy = - labels * torch.log_softmax(logits, dim=-1)
            weighted_entropy = entropy*weights
            loss = loss + weighted_entropy.sum()/self.logit_width

            #Update and smooth weights.
            weights = entropy*(1-self.boost) + self.boost
        return loss