"""
A place to experiment


"""
from __future__ import annotations

import builtins
import os
from functools import partial

import treetensor.torch as torch
import torch as normal

print = partial(builtins.print, sep=os.linesep)

if __name__ == '__main__':
    # create a tree tensor
    t = torch.randn({'a': (2, 3), 'b': {'x': (3, 4)}})
    print(t)
    print(torch.randn(4, 5))  # create a normal tensor
    print()
    print(t.a)

    # structure of tree
    print('Structure of tree')
    print('t.a:', t.a)  # t.a is a native tensor
    print('t.b:', t.b)  # t.b is a tree tensor
    print('t.b.x', t.b.x)  # t.b.x is a native tensor
    print()

    # math calculations
    print('Math calculation')
    print('t ** 2:', t ** 2)
    print('torch.sin(t).cos()', torch.sin(t).cos())
    print()

    # backward calculation
    print('Backward calculation')
    t.requires_grad_(True)
    t.std().arctan().backward()
    print('grad of t:', t.grad)
    print()

    # native operation
    # all the ops can be used as the original usage of `torch`
    print('Native operation')
    print('torch.sin(t.a)', torch.sin(t.a))  # sin of native tensor

    @normal.jit.script
    def retrieve(tensor: torch.Tensor):
        a = tensor.a
        return a

    retrieved = retrieve(t)
