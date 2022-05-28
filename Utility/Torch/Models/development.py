"""
A place to experiment


"""
import torch
from torch import nn
from collections import namedtuple

items = zip(['1', '2,', '3'], [1, 2, 3])
testerclass = namedtuple('tester', ['x', 'y', 'z'])
print(testerclass)
testerinstance = testerclass(x=11, y=22, z=30)
print(testerinstance)
for item in testerinstance:
    print(item)

new_instance = testerinstance._replace(x = 10)
print(new_instance)
print(new_instance._fields)