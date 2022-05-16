








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
                running_stream: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:

        text_stream = self._norm0(text_stream) #Required due to format of stacked layers.

        #Attention and feedforward
        attn = self._attn(text_stream, memory, memory)
        text_stream = self._norm1(attn + text_stream)

        feedforward = self._ff(text_stream)
        text_stream = self._norm2(feedforward + text_stream)
        text_stream = text_stream + running_stream

        #Logits and predictions.
        logits = self._logit(text_stream)
        return logits, text_stream

class ModelReductionEndpoint(nn.Module):
    """
    Creates a single output where previously there were
    many. HIGHLY compatible with unsupervised training techniques.
    """
    def __init__(self, Submodel_Logits: List[nn.Module]):
        """
        :param Submodel_Logits: A collection of submodel
        logits capable of calculating the logits for a
        particular submodel.
        """
        super().__init__()
        self._submodel_logits = Submodel_Logits

    def forward(self, text_streams: List[torch.Tensor], memories: List[torch.Tensor]):
        running_stream = torch.zeros_like(text_streams[0])
        for layer, stream, memory in zip(self._submodel_logits, text_streams, memories):
            _, running_stream = layer(stream, memory, running_stream)
        return running_stream


class ModelPredictionEndpoint(nn.Module):
    """

    Capable of creating the groups of
    predictions utilized in the transformer, then
    using it to make predictions.

    """
    def __init__(self, Submodel_Logits: List[nn.Module]):
        """
        :param vocabulary_size: How big the output vector should be.
        :param Submodel_Logits: A collection of submodel
        logits capable of calculating the logits for a
        particular submodel.
        """
        super().__init__()
        self._submodel_logits = Submodel_Logits

    def forward(self, text_streams: List[torch.Tensor], memories: List[torch.Tensor]):
        running_stream = torch.zeros_like(text_streams[0])
        for layer, stream, memory in zip(self._submodel_logits, text_streams, memories):
            logits, running_stream = layer(stream, memory, running_stream)
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
        running_stream = torch.zeros_like(text_streams[0])
        for layer, stream, memory in zip(self._submodel_logits, text_streams, memories):

            logits, running_stream = layer(stream, memory, running_stream)

            prediction = torch.softmax(logits, dim=-1)
            loss = F.cross_entropy(prediction, labels, weight, reduction="none")
            total_loss = total_loss + loss.mean()

            weight = loss*self._boost_factor

        return total_loss, prediction, logits