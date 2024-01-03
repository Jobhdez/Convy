import torch
import torch.nn as nn
from compiler import  get_node_inputs

class VGG16Block(nn.Module):
    def __init__(self):
        super(VGG16Block, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x
net = VGG16Block()

# Creating an example input tensor
batch_size = 1
channels = 3  # RGB channels
height = 224
width = 224

example_input = torch.randn(batch_size, channels, height, width)

module = torch.jit.trace(net, example_input)

state_dict = module.state_dict()

graph = module.graph.copy()

nodes = list(graph.nodes())

### Example

inputs = get_node_inputs(module)
print(inputs)

### ->
"""
['__torch__.VGG16Block', '__torch__.VGG16Block', '__torch__.VGG16Block', '__torch__.torch.nn.modules.conv.___torch_mangle_7.Conv2d', 'Float(1, 3, 224, 224, strides=[150528, 50176, 224, 1], requires_grad=0, device=cpu)', '__torch__.torch.nn.modules.batchnorm.BatchNorm2d', 'Tensor', '__torch__.torch.nn.modules.activation.ReLU', 'Tensor']"""
