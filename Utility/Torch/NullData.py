"""

Designing a layer for a particular task is hard.
This section provides random data to make it easier.

1) Generate a task appropriate null set
2) Train on it. Watch the loss
3) Compare the loss behavior when beginning training to the loss behavior of a control layer,
    usually just a dense selector.
4) If loss drops off sharply near the beginning, before leveling out as memorization begins,
    you have a task improvement.

"""

import torch
from torch import nn

class Catagorical_Reduction_Embedding():
    """
    Makes data for catagorical embedding mapping. This
    consists of mapping a large collection of inputs to
    a much smaller collection of outputs. Generates batches
    and labels upon demand.

    Output is placed in the batch dimension of a transformer stream,
    with the items channel having a width of 1. Prediction should be logit
    of width map_options.
    """
    def loss(self, prediction, labels):
        return self.loss(prediction, labels)

    def __init__(self, embedding_width: int, samples: int, map_options: int, batch_width: int):
        labels = torch.arange(0, map_options)
        labels = torch.multinomial(labels.type(torch.float), samples, replacement=True).type(torch.long)
        inputs = torch.randn([samples, 1, embedding_width])

        self.sample_options = samples
        self.batch_width = batch_width
        self.labels = labels
        self.inputs = inputs
        self.loss = nn.CrossEntropyLoss()
    def __call__(self):
        batch_choices = torch.randint(0, self.sample_options, [self.batch_width])
        batch_labels = self.labels[batch_choices]
        batch_data = self.inputs[batch_choices, :, :]
        return batch_data, batch_labels

class SimpleGameMovementEmbedding():
    """
    An N turn game process. An initial game
    state, in which a onehot encoding exists
    with one operation hot, is provided. A
    hidden entity, known as the state update,
    is applied to this N times and deterministically
    places the output into a new location.

    The task is then to find out the final position N turns later.






    """
    def hash_it(self, tensor: torch.Tensor):
        for _ in range(self.iterations):
            tensor = self.hash_iteration(tensor)
        return tensor
    def hash_iteration(self, tensor: torch.Tensor):
        output = torch.matmul(self.weight, tensor) + self.bias
        output[::2] = -output[::2]
        return output
    def loss(self, prediction, labels):
        hash = self.hash_it(prediction)
        loss = self.loss(hash, labels)
        return loss
    def __init__(self,  embedding_width: int,  hash_iterations: int, batch_size: int):
        self.weight = torch.randn([embedding_width, embedding_width])
        self.bias = torch.randn([embedding_width])
        self.iterations = hash_iterations
        self.batch_size = batch_size
        self.loss = nn.CosineSimilarity()
    def __call__(self):
        hash_input = torch.normal(1, 1, [self.batch_size])
        labels = self.hash_it(hash_input)
        return labels



