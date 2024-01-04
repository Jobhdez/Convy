import torch
import torch.nn as nn
from compiler import  get_node_inputs
from conv2d import convolution
from torch.nn import functional as F
from torch.jit.annotations import Optional
import numpy as np

class VGG16Block(nn.Module):
    def __init__(self):
        super(VGG16Block, self).__init__()
        self.conv1 = nn.Conv2d(1,1,3)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

"""
   the following functions, namely `convolution_torch`, `batch_norm2d`, `rectified`,
   correspond to the forward pass of the above network `VGG16Block`.

   the idea is that once the compiler has the input tensor data, and identifies the conv2d and batchnorm2d and
   the relu layers, it can then generate the C or Cuda call. So the implementations of these functions below
   will help me generate the C code.

"""

def convolution_torch(input_data, weight, bias):
    # Assuming 'input_data' is a 4D tensor (batch_size, channels, height, width)
    _, _, input_height, input_width = input_data.size()
    _, _, filter_height, filter_width = weight.size()

    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1

    output = torch.zeros(1, 1, output_height, output_width)

    for h in range(output_height):
        for w in range(output_width):
            receptive_field = input_data[:, :, h:h+filter_height, w:w+filter_width]
            output[:, :, h, w] = torch.sum(receptive_field * weight) + bias

    return output

def batch_norm2d(input_data, gamma, beta, epsilon=1e-5):
    # Calculate mean and variance along batch and spatial dimensions
    mean = torch.mean(input_data, dim=(0, 2, 3), keepdim=True)
    var = torch.var(input_data, dim=(0, 2, 3), unbiased=False, keepdim=True)

    # Normalize input_data
    normalized_data = (input_data - mean) / torch.sqrt(var + epsilon)

    # Scale and shift
    output_data = gamma * normalized_data + beta

    return output_data

def rectified(x):
    return torch.max(torch.tensor(0.0), x)


"""
make scriptmodule, i.e. torch.jit.trace(..)
"""

net = VGG16Block()


example_forward_input = torch.randn(1, 1, 224, 224) 
module = torch.jit.trace(net, example_forward_input)

state_dict = module.state_dict()


"""
== Example ==

Run example a vgg block being computed  with custom functions and comparing to the actual result
from pytorch.
"""

def run_vgg_example(input_tensor,
                    conv_weight,
                    conv_bias,
                    bn_weights,
                    bn_biases):
    print("running example ...")

    x = convolution_torch(example_forward_input_2, conv_weights, conv_biases)
    x = batch_norm2d(x, bn_weights, bn_biases)
    x = rectified(x)
    savm = module.save("test26.pth")
    loadm = torch.jit.load("test26.pth")

    with torch.no_grad():
        torch_output = loadm(example_forward_input_2)

    return f'torch output: {torch_output} \n\n myoutput: {x}'



conv_name = 'conv1'
bn_name = 'bn1'

conv_weights = state_dict[conv_name + '.weight']
conv_biases = state_dict[conv_name + '.bias']


bn_weights = state_dict[bn_name + '.weight']
bn_biases = state_dict[bn_name + '.bias']

bn_running_mean = state_dict[bn_name + '.running_mean']
bn_running_var = state_dict[bn_name + '.running_var']

example_forward_input_2 = torch.randn(1, 1, 224, 224)

run_vgg_example(example_forward_input_2, conv_weights, conv_biases, bn_weights, bn_biases)


##---------------------------------------
##--------------torch.fx-----------------

from torch.fx.node import Node
from typing import Dict
import torch

class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape, element type,
    and tensor data for the output values of each operation on
    the `shape`, `dtype`, and `data` attributes of the operation's
    `Node`.
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype
                node.data = result
                

            env[node.name] = result

        return load_arg(self.graph.result)

from torch.fx import symbolic_trace
gm = symbolic_trace(net)

#input_tensor = torch.randn(1,1,32,32)
#result = sp.propagate(input_tensor)

gmodule = torch.fx.GraphModule(net, gm.graph)

# Create an instance of ShapeProp using the GraphModule
shape_prop = ShapeProp(gmodule)

# Example input tensor
input_tensor = torch.randn(1, 1, 32, 32)

# Propagate shapes through the GraphModule
result = shape_prop.propagate(input_tensor)
