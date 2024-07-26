import torch
import torch.nn as nn
#from compiler import  get_node_inputs
#from conv2d import convolution
from torch.nn import functional as F
from torch.jit.annotations import Optional
import numpy as np
#from src.frontend.torchfx import ShapeProp, get_layers

import torch
import torch.nn as nn
import torch.fx as fx

# Define the VGG16Block class
class VGG16Block(nn.Module):
    def __init__(self):
        super(VGG16Block, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

# Create an instance of the class
net = VGG16Block()

# Create an input tensor
input_tensor = torch.randn(1, 1, 5, 5)  # Example input tensor with shape (batch_size, channels, height, width)

# Trace the module to get the GraphModule
gm = fx.symbolic_trace(net)

# Print the GraphModule

# Inspect the graph to extract input tensor name, weights, and biases
for node in gm.graph.nodes:
    print(f"Node op: {node.op}, name: {node.name}, target: {node.target}, args: {node.args}, kwargs: {node.kwargs}")

    if node.op == 'placeholder':
        print(f"Input tensor name: {node.name}")
    elif node.op == 'call_module':
        submod = dict(gm.named_modules())[node.target]
        if isinstance(submod, nn.Conv2d):
            print(f"Conv2d weights: {submod.weight}")
            print(f"Conv2d bias: {submod.bias}")
        elif isinstance(submod, nn.BatchNorm2d):
            print(f"BatchNorm2d weights: {submod.weight}")
            print(f"BatchNorm2d bias: {submod.bias}")
        elif isinstance(submod, nn.ReLU):
            print(f"ReLU has no weights or bias")

# Run the input tensor through the traced graph
#output_tensor = gm(input_tensor)
#print(f"Output tensor: {output_tensor}")

# running python vgg_example.py gives:

"""
(compiler) [lara@manifold examples]$ python vgg_example.py 
Node op: placeholder, name: x, target: x, args: (), kwargs: {}
Input tensor name: x
Node op: call_module, name: conv1, target: conv1, args: (x,), kwargs: {}
Conv2d weights: Parameter containing:
tensor([[[[-0.0778, -0.2790, -0.2657],
          [-0.1351,  0.0044,  0.2237],
          [-0.1920, -0.0218,  0.1268]]]], requires_grad=True)
Conv2d bias: Parameter containing:
tensor([0.0631], requires_grad=True)
Node op: call_module, name: bn1, target: bn1, args: (conv1,), kwargs: {}
BatchNorm2d weights: Parameter containing:
tensor([1.], requires_grad=True)
BatchNorm2d bias: Parameter containing:
tensor([0.], requires_grad=True)
Node op: call_module, name: relu1, target: relu1, args: (bn1,), kwargs: {}
ReLU has no weights or bias
Node op: output, name: output, target: output, args: (relu1,), kwargs: {}
"""
"""
note: the input tensor you can manipulate outside of all this because
its the input"""
