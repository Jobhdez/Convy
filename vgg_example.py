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
net = VGG16Block()


example_forward_input = torch.randn(1, 1, 224, 224) 
module = torch.jit.trace(net, example_forward_input)

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

import torch

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


# relu
def rectified(x):
    return torch.max(torch.tensor(0.0), x)

"""
batch_size = 1
channels = 3  # RGB channels
height = 224
width = 224
input_tensor = torch.randn(batch_size, channels, height, width)
input_tensor_2d = input_tensor.view(batch_size, channels, -1)
in_channels = channels
out_channels = 64
kernel_size = (3, 3)
"""
new_input_data = np.ones((1, 1, 224, 224), dtype=np.float32)
ones_pytorch = torch.tensor(new_input_data, dtype=torch.float32)
conv_name = 'conv1'
bn_name = 'bn1'

# Extract convolutional layer weights and biases
conv_weights = state_dict[conv_name + '.weight']
conv_biases = state_dict[conv_name + '.bias']

# Extract batch normalization layer weights, biases, running mean, and running variance
bn_weights = state_dict[bn_name + '.weight']
bn_biases = state_dict[bn_name + '.bias']
bn_running_mean = state_dict[bn_name + '.running_mean']
bn_running_var = state_dict[bn_name + '.running_var']

""""
x = conv2d(inp, weight, bias)
#x = convolution(input_tensor
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
"""
x = convolution_torch(ones_pytorch, conv_weights, conv_biases)
#x,_,_ = batch_norm(x, bn_weights, bn_biases, bn_running_mean, bn_running_var, eps=1e-5, momentum=0.1)
#print(x)
m = nn.BatchNorm2d(1)
x = m(x)
x = rectified(x)
savm = module.save("test25.pth")
loadm = torch.jit.load("test25.pth")

with torch.no_grad():
    torch_output = loadm(ones_pytorch)


print(f'torch output: {torch_output} \n\n myoutput: {x}')


                
