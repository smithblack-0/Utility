import torch
from torch import nn
import uuid



lass ActionEntity():
    """
    
    Found within an actor, and elsewhere.
    Maintains a database of deletion callbacks.
    
    """



class Actor(nn.Module):
    """

    A class which can produce a
    policy - a set of unique id's
    used to indicate something downstream.


    """

    pass

class MultiCategoricalActor(Actor):
    """ Makes a choice, and then generates an action based on it."""

    def __init__(self, seed_width: int, threshold: float = 0.1,
                 trim_percentage: float = 0.6, expand_percent: float = 0.9):

        super().__init__()
        #Initialize the actions, logits, and coefficient
        actions = torch.stack([torch.tensor(uuid.uuid1()) for _ in range(seed_width)], dim=-1)
        logits = -torch.log(torch.tensor(seed_width, dtype=torch.float32)) * torch.ones(seed_width)

        #Register these as buffers.
        self.actions = self.register_buffer('actions', actions)
        self.logits = self.register_buffer('logits', logits)
        self.coefficient = self.register_buffer('coefficient', coefficient)
        self.id = self.register_buffer(torch.tensor(uuid.uuid1()))
        self.threshold = self.register_buffer(torch.tensor(threshold))

    def forward(self):

        #Append on the generation probability. Then perform sample
        threshold = torch.softmax(self.logits)*torch.randn(self.logits.shape[0]) > self.threshold
        sample = self.logits.masked_select(threshold)









