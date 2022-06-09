import copy
from typing import Optional, Tuple, NamedTuple, List, Callable

import torch
from torch import nn
from torch.nn import functional as F

from Utility.Torch.Models.SupertransformerOld import StreamTools
from Utility.Torch.Models.SupertransformerOld.StreamTools import StreamTensor
from Utility.Torch.Models.SupertransformerOld.EnsembleTools import Scheduler

"""

The modules in this class are responsible for taking tensors developed by the individual 
subensembles and using them to make predictions. The ensemble stream, with friends, is 
handed to the teardown 

"""



class AbstractTeardown(nn.Module):
    """
    A class responsible for finishing up with
    an ensemble of some sort, producing a prediction stream and ensemble stream.

    -- forward params --

    ensemble_stream: The item from the currently evaluated ensemble.
    cumulative_stream: The item from the last evaluated ensemble. Optional.
    auxiliary_stream: Any additional information the user may want, such as losses
        or training commands. Optional.
    """
    def __init__(self):
        super().__init__()
    def forward(self,
                ensemble_stream: StreamTensor,
                cumulative_stream: StreamTensor,
                auxiliary_stream: NamedTuple) \
            -> Tuple[StreamTensor, StreamTensor, StreamTensor]:
        """

        :param ensemble_stream: The current stream from the current instance
        :param recursive_stream: Things found or placed here will tend to persist.
        :param cumulative_stream: The cumulative stream. From the previous iteration.
        :param auxiliary_stream: Anything from the auxiliary stream
        :return: Three items. First, the cumulative stream tensor. Second, the ensemble stream. Third, the
        recursion stream.
        """
        raise NotImplementedError("Must impliment forward function in EnsembleTeardown")


class CategoricalBoostedPredictor(AbstractTeardown):
    """
    A teardown designed to work with categorical data, providing the losses and predictions
    as an output. Performs boosting - each step updates the previous guess, as in XGBOOST,
    and the result is accumulated along the cumulative channel. The incoming ensemble
    stream is routed to the channel output.

    This class has provisions for scheduling. The two schedules available
    are smoothing scheduling, and engagement scheduling. smoothing scheduling
    must return a value between 0 and 1, where 0 is no lable smoothing, 1 is complete.
    Engagement must return a value between 0 and 1, where 0 is no contribution to output,
    and 1 is completely on.
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
                 label_name: str,
                 index_name: str = 'index',
                 smoothing_scheduler: Optional[Scheduler] = None,
                 engagement_scheduler: Optional[Scheduler] = None,
                 ):
        """
          The channel name should be the channel to boost catagorize. It should be a string.\n
        The smoothing scheduler is a callable. It should accept a scalar, the


        :param input_width: The width of the input channel
        :param logit_width: The width of the logits, or number of catagories
        :param channel_name: The name of the channel to work with
        :param label_name: The name of the labels. Found in the auxiliary stream.
        :param index_name: The name the index is found under
        :param smoothing_scheduler: A scheduler. What the smoothing should look like.
        :param engagement_scheduler: A scheduler. What the engagement should look like.
        """

        super().__init__()


        self.logits_name = channel_name + ' logits'
        self.lossname = channel_name + ' loss'
        self.channel_name = channel_name

        self.index_name = index_name

        self.logits = nn.Linear(input_width, logit_width)
        self.labels = label_name

        self.smoothing_scheduler = smoothing_scheduler
        self.engagement_scheduler = engagement_scheduler

        self.batch_counter = 0
    def forward(self,
                ensemble_stream: StreamTools.StreamTensor,
                recursive_stream: StreamTensor,
                cumulative_stream: StreamTensor,
                auxiliary_stream:  NamedTuple)\
            -> Tuple[StreamTensor, StreamTensor, StreamTensor]:

        #Isolate required information from stream. Then calculate logits
        (tensor) = ensemble_stream.isolate([self.channel_name])

        labels = auxiliary_stream.labels
        index = auxiliary_stream.channel
        epoch = auxiliary_stream.epoch
        cumulative_batch = auxiliary_stream.cumbatch

        logits = self.logits(tensor)

        # Get schedule. Smoothing should be between 0 and 1, where 1 menas completely
        #smoothed. Engagement should be between 0 and 1, where 1 means completely engaged

        if self.smoothing_scheduler is not None:
            smoothing = self.smoothing_scheduler(logits, labels, self.batch_counter, index)
        else:
            smoothing = 0.0

        if self.engagement_scheduler is not None:
            engagement = self.engagement_scheduler(logits, labels, self.batch_counter, index)
        else:
            engagement = 1.0

        assert 0 <= engagement <= 1
        assert 0 <= smoothing <= 1

        #Update best guess. Get weights
        if self.logits_name in cumulative_stream.stream:
            (cumulative_tensor) = cumulative_stream.isolate([self.logits_name])
        else:
            cumulative_tensor = torch.zeros_like(logits)

        #Create final logits by combining ensemble logits with cumulative logits, keeping in mind the
        #specifications defined by engagement. Then compute loss and new weights.

        logits = logits*engagement + cumulative_tensor
        loss = F.cross_entropy(logits, labels, label_smoothing=smoothing)
        reduced_loss = loss.mean()

        ##Create the stream update, with metrics included.

        stream_items = {self.logits_name: logits}
        loss_items = {self.lossname: reduced_loss}
        metric_items = {self.lossname: [loss]}
        update_stream = StreamTensor(stream_items, loss_items, metric_items)

        #If cumulative stream was provided, merge the cumulative stream and update. Else,
        #start the cumulative stream.
        null_stream = cumulative_stream.keeponly([])
        merger = StreamTools.StreamMerger([null_stream, update_stream])
        merger.stream.sum()
        merger.losses.sum()
        output = merger.build()

        #Rebuild the ensemble stream with the new logit information included. Retjurn

        insights = StreamTensor({'logits' : logits, 'labels' : labels} )

        self.batch_counter = self.batch_counter + 1
        return output, ensemble_stream.clone(True, False, False), insights


class PredictorCollection(nn.Module):
    """
    A holder for a group of predictors, which will
    interact correctly with an oncoming ensemble stream.

    methods:

    append: Appends a new predictor to the collection
    generate: Generate a collection based on the provided predictor class, and
        the arguments to build with.
    """
    @staticmethod
    def make(predictor: AbstractPredictor, number: int, *args, **kwargs):
        """
        Using a class instance, makes a collection of number predictors.

        :param predictor:
        :param number:
        :param args:
        :param kwargs:
        :return:
        """

    def __init__(self, predictors: Optional[List[AbstractPredictor]] = None):
        super().__init__()
        self.prediction_instances: List[AbstractPredictor] = []
        if predictors is not None:
            self.prediction_instances = self.prediction_instances + predictors
    def forward(self,
                ensemble_streams: List[StreamTensor],
                recursive_streams: List[StreamTensor],
                auxiliary_streams: List[NamedTuple],
                ):
        cumulative_stream: StreamTensor = StreamTensor()
        iterator = zip(ensemble_streams, recursive_streams, auxiliary_streams, self.prediction_instances)
        for ensemble, recursive, auxiliary_stream, instance in iterator:
            instance(ensemble, recursive, cumulative_stream, auxiliary_stream)




