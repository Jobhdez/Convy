import torch
import torch.nn as nn
from compiler import  get_node_inputs

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.jit.annotations import Optional

class VGG16Block(nn.Module):
    def __init__(self):
        super(VGG16Block, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
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

inp2  = torch.tensor([[0.0,3.0,2.0], [3.0, 4.5, 5.6], [5.6, 6.0,5.0]])
example_input = inp2.view(1, 1, inp2.size(0), inp2.size(1))
module = torch.jit.trace(net, example_input)

# Rest of your code...

state_dict = module.state_dict()

graph = module.graph.copy()

nodes = list(graph.nodes())

### Example

inputs = get_node_inputs(module)
print(inputs)

### ->
"""
['__torch__.VGG16Block', '__torch__.VGG16Block', '__torch__.VGG16Block', '__torch__.torch.nn.modules.conv.___torch_mangle_7.Conv2d', 'Float(1, 3, 224, 224, strides=[150528, 50176, 224, 1], requires_grad=0, device=cpu)', '__torch__.torch.nn.modules.batchnorm.BatchNorm2d', 'Tensor', '__torch__.torch.nn.modules.activation.ReLU', 'Tensor']"""

### as of jan 2 the code below doesnt work :)

### Batchnorm, taken from here
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def conv2d(input_tensor, weight, bias):
    


    corr = corr2d(input_tensor, weight)
    return corr + bias


# relu
def rectified(x):
    return torch.max(torch.tensor(0.0), x)


batch_size = 1
channels = 3  # RGB channels
height = 224
width = 224
input_tensor = torch.randn(batch_size, channels, height, width)
input_tensor_2d = input_tensor.view(batch_size, channels, -1)
in_channels = channels
out_channels = 64
kernel_size = (3, 3)

# Initialize weight tensor
weight = nn.Parameter(torch.rand(kernel_size))
bias = nn.Parameter(torch.zeros(1))
inp = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

x = conv2d(inp, weight, bias)
print(x)

shape = (1, 64, 1, 1)
gamma = nn.Parameter(torch.ones(shape))
beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
moving_mean = torch.zeros(shape)
moving_var = torch.ones(shape)
x, _, _ = batch_norm(x, gamma, beta,
                             moving_mean, moving_var,
                            eps=1e-5, momentum=0.1)
x = rectified(x)
print(x)
savm = module.save("test25.pth")
loadm = torch.jit.load("test25.pth")

with torch.no_grad():
    inp = inp.view(1, 1, inp.size(0), inp.size(1))
    torch_output = loadm(inp)


print(f'torch output: {torch_output} \n\n myoutput: {x}')
