"""
A place to experiment


"""
import torch
from torch import nn

dim = 4
vector = torch.ones([4])
matrix = torch.randn([dim, dim])
orthogonal = torch.nn.init.orthogonal_(matrix)

item = torch.matmul(orthogonal, vector)
item[0] = 0
item2 = torch.matmul(orthogonal.transpose(-1, -2), item)
print(matrix)
print(orthogonal)

print(vector)
print(item)
print(item2)