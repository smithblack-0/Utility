from typing import List, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from Utility.Torch.Learnables import Layers


"""

The text flow encoding level. This is fairly comparible
to modern NPL machine learning algorithms, though with a 
little extra depth. 

DIAGRAMS:

https://docs.google.com/drawings/d/1Ej0ZlPbTqyDC_aC1xiMwghGn28IDDC8667dgyHr-I0Y/edit


TERMINOLOGY:


** Architecture **

Stream_Submodel: A single stream of predicitive information traveling from the source data, 
    accepting prior residuals, and returning processed information
FlowExchange: The process of exchanging information between the memory and stream tensors
within a submodel, while doing stream level processing in between.
ResidualBypass: The process of reinserting individual sublayer outputs into the next
sublayer input.


** Tensors **

TextStream: A text based embedding of some sort. Arbitrary length
Memory: A tensor of some sort. Known length. Attends to an entire text stream

FEATURES:

- Effective depth

First, many parallel stacks of the same sequence of model exists,
with residual redirect connecting them. Data may take as short,
 or as long, a path as it wishes before reaching the end,
 avoiding a large portion of the dead gradient problems by ensuring
 the distance between a calculation which has drawn a conclusion,
 and the exit point, is very short.

- Boosting and Pretraining

The existance of many parallel stacks allows the usage of a useful technique - boosting.
During pretraining, it is possible to place a projection stack on the end of each individual
submodel, and check how far off from true the particular example is. 

"""

class Interchange(nn.Module):
    """

    A memory exchange layer. Accepts a
    text stream and memory argument. Returns
    a new text_stream and memory argument.

    Itself responsible Principly for memory exchange
    and outer normalization. Provided to it will be
    layer stacks responsible for attention and feedforward steps.

    Internal flow according to:
    https://docs.google.com/drawings/d/1Ej0ZlPbTqyDC_aC1xiMwghGn28IDDC8667dgyHr-I0Y/edit
    """

    def __init__(self,
                 d_stream: int,
                 d_memory: int,
                 stream_processing: nn.Module,
                 memory_processing: nn.Module,
                 transfer_heads: int = 2,
                 dropout: float = 0.1,
                 ):
        """

        :param d_stream: Width of stream embeddings
        :param d_memory: Width of memory embeddings
        :param stream_processing: A module which handles anything like self attn, conv, and feedforward.
        :param memory_processing: A module which handles self attn and feedforward
        :param transfer_heads: How many heads the interchange has. Default of 2
        :param dropout: Dropout rate. default of 0.1
        """
        super().__init__()

        # Store processing
        self._stream_processing = torch.jit.script(stream_processing)
        self._memory_processing = torch.jit.script(memory_processing)

        # Create norms
        self._stream_intake_norm = nn.LayerNorm(d_stream)
        self._memory_intake_norm = nn.LayerNorm(d_memory)

        self._stream_output_norm = nn.LayerNorm(d_stream)
        self._memory_output_norm = nn.LayerNorm(d_memory)

        # Create exchange attentions.
        self._stream_to_memory_attn = nn.MultiheadAttention(d_memory,
                                                            transfer_heads,
                                                            dropout,
                                                            kdim=d_stream,
                                                            vdim=d_stream)
        self._memory_to_stream_attn = nn.MultiheadAttention(d_stream,
                                                            transfer_heads,
                                                            dropout,
                                                            kdim=d_memory,
                                                            vdim=d_memory)

    def forward(self, text_stream: torch.Tensor, memory: torch.Tensor, args: List[torch.Tensor]):

        # Normalize input stream, then transfer stream info into memory.
        text_stream = self._stream_intake_norm(text_stream)
        attn = self._stream_to_memory_attn(memory, text_stream, text_stream)
        memory = self._memory_intake_norm(attn + memory)

        # Perform processing, including layernorm. Include residual bypass
        if len(args) > 0:
            text_stream = self._stream_output_norm(self._stream_processing(text_stream, *args) + text_stream)
            memory = self._memory_output_norm(self._memory_processing(memory, *args) + memory)
        else:
            text_stream = self._stream_output_norm(self._stream_processing(text_stream) + text_stream)
            memory = self._memory_output_norm(self._memory_processing(memory))

        # Transfer memory information into stream, then return values
        text_stream = self._memory_to_stream_attn(text_stream, memory, memory)
        return text_stream, memory

class Stream_Submodel(nn.Module):
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



class Stream_Model(nn.Module):
    """

    Creates memory starts. Goes through memory layer sequences.
    Stores and then uses residuals when given to improve performance.
    Starts residual chain as needed. Creates oracle layer inputs.

    """

    class Seed(nn.Module):
        """

        A small container to hold seed values. Used because
        torchscript does not like parameterlists, and I design
        code to be savable.

        """

        def __init__(self, shape, dtype):
            super().__init__()
            seed = torch.zeros(shape, dtype=dtype, requires_grad=True)
            torch.nn.init.kaiming_uniform_(seed)
            self._seed = nn.Parameter(seed)

        def forward(self):
            return self._seed

    def __init__(self,
                 d_memory: int,
                 memory_width: int,
                 stream_submodels: List[nn.Module],
                 calibration_layer: nn.Module,
                 dtype):
        """
        :param d_memory: The encoding width of the memory layers
        :param memory_width: How wide the memory bus should be, per superlayer
        :param stream_submodels: The submodels to use in processing the stream flow.
        """

        super().__init__()

        # Create the memory seeds
        memory = nn.ModuleList()
        for _ in stream_submodels:
            memory.append(self.Seed([memory_width, d_memory], dtype))

        # store
        self._submodels = stream_submodels
        self._memory_seeds = memory


    def forward(self,
                text_stream: torch.Tensor,
                args: List[torch.Tensor]) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor]]:



        residuals: List[torch.Tensor] = []
        outputs: List[torch.Tensor] = []
        memories: List[torch.Tensor] = []
        for layer, seed in zip(self._submodels, self._memory_seeds):
            memory = seed()
            stream = text_stream
            output, memory, new_residuals = layer(stream, memory, residuals, args)

            outputs.append(output)
            memories.append(memory)
            residuals = new_residuals
        return outputs, memories

### Endpoint processes ###


class SubmodelLogits(nn.Module):
    """
    Given the prior information updates the current guess, and the current
    logits.

    It folds the memory information into the text stream using an
    encoder layer, then predicts logits while incorporating the previous guess.
    """

    def __init__(self,
                 d_stream: int,
                 d_memory: int,
                 heads: int,
                 vocabulary_size: int):
        super().__init__()

        self._vocab_size = vocabulary_size

        self._attn = nn.MultiheadAttention(d_stream, heads, kdim=d_memory, vdim=d_memory)
        self._ff = Layers.FeedForward(d_stream, d_stream*2)
        self._logit = nn.Linear(d_stream, vocabulary_size)
        self._norm0 = nn.LayerNorm(d_stream)
        self._norm1 = nn.LayerNorm(d_stream)
        self._norm2 = nn.LayerNorm(d_stream)

    def forward(self,
                text_stream: torch.Tensor,
                memory: torch.Tensor,
                running_logits: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:

        text_stream = self._norm0(text_stream) #Required due to format of stacked layers.

        #Attention and feedforward
        attn = self._attn(text_stream, memory, memory)
        text_stream = self._norm1(attn + text_stream)

        feedforward = self._ff(text_stream)
        text_stream = self._norm2(feedforward + text_stream)

        #Logits and predictions.
        logits = self._logit(text_stream)
        logits = logits + running_logits

        return logits



class ModelPredictionEndpoint(nn.Module):
    """

    Capable of creating the groups of
    predictions utilized in the transformer, then
    using it to make predictions.

    """
    def __init__(self,
                 vocabulary_size: int,
                 Submodel_Logits: List[nn.Module]):
        """
        :param vocabulary_size: How big the output vector should be.
        :param Submodel_Logits: A collection of submodel
        logits capable of calculating the logits for a
        particular submodel.
        """
        super().__init__()
        self._seed_shape = torch.zeros([vocabulary_size], dtype=torch.float32, requires_grad=False)
        self._submodel_logits = Submodel_Logits

    def forward(self, text_streams: List[torch.Tensor], memories: List[torch.Tensor]):

        logits = self._seed_shape
        for layer, stream, memory in zip(self._submodel_logits, text_streams, memories):
            logits = layer(stream, memory, logits)
        predictions = torch.softmax(logits, dim=-1)
        return predictions, logits


class ModelTrainingEndpoint(nn.Module):

    """
    A loss endpoint for running boost training logic.

    Returns a loss, not a prediction. Performs boosted
    loss generation, where each layer's results
    are added to the previous, predictions are made,
    and the wrong results are emphasized. Accumulates
    a loss function, and returns.

    """

    def __init__(self,
                 vocabulary_size: int,
                 Submodel_Logits: List[nn.Module],
                 boost_factor: float = 2):
        """
        :param vocabulary_size: How big the output vector should be.
        :param Submodel_Logits: A collection of submodel
        logits capable of calculating the logits for a
        particular submodel.
        :param boost_factor: How strongly the boosting will apply
        """
        super().__init__()
        self._seed = torch.zeros([vocabulary_size], dtype=torch.float32, requires_grad=False)
        self._submodel_logits = Submodel_Logits
        self._boost_factor = boost_factor

    def forward(self, text_streams: List[torch.Tensor], memories: List[torch.Tensor], labels):

        total_loss = torch.tensor(0.0)
        weight = torch.ones([text_streams[0].shape[-1]], dtype=torch.float32)
        logits = self._seed
        prediction = torch.softmax(logits, dim=-1)
        for layer, stream, memory in zip(self._submodel_logits, text_streams, memories):

            logits = layer(stream, memory, logits)

            prediction = torch.softmax(logits, dim=-1)
            loss = F.cross_entropy(prediction, labels, weight, reduction="none")
            total_loss = total_loss + loss.mean()

            weight = loss*self._boost_factor

        return total_loss, prediction, logits


