import torch
import torch.nn as nn
from conv2d import get_nodes, get_inputs
import re

def compile(module):
    graph = module.graph.copy()
    nodes = get_nodes(graph)
    inputs = []
    for node in nodes:
        input_nodes = list(node.inputs())
        inputs.append(input_nodes)

    for node in inputs:
        for inp in node:
            match inp.type().str():
                case '__torch__.conv2d.Net':
                    state_dict = module.state_dict()
                    weight = state_dict['conv.weight']
                    bias = state_dict['conv.bias']
                    device = re.findall('device=(cpu|gpu)', inp.__repr__())
                    device = device[0]
                    conv2d = Conv2d(weight, bias, device)

                    return conv2d

                case _:
                    break

    
class Conv2d:
    def __init__(self, weight, bias, device):
        self.weight = weight
        self.bias = bias
        self.device = device

"""
>>> from compiler import compile 
torch output: tensor([[[[0.4093]]]]) 

 numpy_output: [[[[0.40930814]]]]
>>> from conv2d import module
>>> ll = compile(module)
>>> ll
<compiler.Conv2d object at 0x7ff4b3672c50>
>>> ll.device 
'cpu'
>>> ll.weight 
tensor([[[[ 0.0225,  0.2030,  0.1440],
          [-0.1560,  0.1994, -0.3088],
          [-0.2562,  0.2457,  0.0221]]]])
>>> ll.bias
tensor([0.2936])
>>>
"""

def get_node_inputs(module):
    graph = module.graph.copy()
    nodes = get_nodes(graph)
    inputs = []
    for node in nodes:
        input_nodes = list(node.inputs())
        inputs.append(input_nodes)

    input_types = []
    for node in inputs:
        for inp in node:
            input_types.append(inp.type().str())

    return input_types

def get_nodes(graph):
    return [node for node in graph.nodes()]

def get_operations(module):
    graph = module.graph.copy()
    nodes = get_nodes(graph)
    
    operations = []
    for node in nodes:
        operation_type = node.kind()
        operations.append(operation_type)

    return operations

