import torch
from torch import nn
from torch.nn import functional as F


def manual_crossentropy(y, labels):
    weights = labels*torch.log_softmax(y, dim=-1)
    loss = -weights.mean()/labels.shape[0]
    return loss

input = torch.randn([20, 10])
labels = torch.randint(9, [20])
labels = F.one_hot(labels, 10).type(torch.float32)

print(F.cross_entropy(input, labels))
print(manual_crossentropy(input, labels))

