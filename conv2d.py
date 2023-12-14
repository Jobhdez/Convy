import torch
import torch.nn as nn
import numpy as np
import numpy

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3).float()

    def forward(self, x):
        return self.conv(x)

n = Net()
example_weight = torch.rand(1, 1, 3, 3)
example_forward_input = torch.rand(1, 1, 3, 3)

# Trace a specific method and construct `ScriptModule` with
# a single `forward` method
###module = torch.jit.trace(n.forward, example_forward_input)

# Trace a module (implicitly traces `forward`) and construct a
# `ScriptModule` with a single `forward` method
module = torch.jit.trace(n, example_forward_input)



def get_params(module):
    params = module.state_dict()
    keys = params.keys()
    vals = params.values()

    return dict(zip(keys, vals))

def get_nodes(graph):
    return list(graph.nodes())

def get_inputs(graph):
    inputs = [i for i in graph.inputs()]

    for i in inputs:
        match i.type():
            case torch._C.TensorType():
                sizes = i.type().sizes()
                strides = i.type().strides()
                dtype = i.type().dtype()
                dim = i.type().dim()
                device = i.type().device()

                return (sizes, strides, dtype, dim, device)

            case _:
                raise ValueError(f'{i} is not supported.')


state_dict = module.state_dict()
# Manually set the weights and biases
weight = state_dict['conv.weight'].numpy()
bias = state_dict['conv.bias'].numpy()

# Manually perform the convolution operation
def convolution(input_data, weight, bias):
    # Assuming 'input_data' is a 4D tensor (batch_size, channels, height, width)
    _, _, input_height, input_width = input_data.shape
    _, _, filter_height, filter_width = weight.shape

    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1

    output = np.zeros((1, 1, output_height, output_width))

    for h in range(output_height):
        for w in range(output_width):
            receptive_field = input_data[:, :, h:h+filter_height, w:w+filter_width]
            output[:, :, h, w] = np.sum(receptive_field * weight) + bias

    return output

# Create a new input tensor
new_input_data = np.random.rand(1, 1, 3, 3)

# Perform the convolution manually
output = convolution(new_input_data, weight, bias)

#print("Model Output:\n", output)

########################
# 
# Here I will be comparing with the prediction from pytorch and my soon to be generated convolution kernel
# this example illustrates how  a deep learning compiler works, kind of
# we will use the weights from `module.state_dict()` above and the bias of a trained network.
# from this we will generate efficient gpu code for the operators which im pretty sure corresponds to the
# forward pass of the pytorch model definition.
# we then apply the generated code for the operators on new inputs to do the inference.
#
# Examples:
#######################

ones = np.ones((1, 1, 3, 3), dtype=np.float32)

ones_pytorch = torch.tensor(ones, dtype=torch.float32)

savm = module.save('test1.pth')
loadm = torch.jit.load('test1.pth')

with torch.no_grad():
    torch_output = loadm(ones_pytorch)

np_output = convolution(ones, weight, bias)

print(f'torch output: {torch_output} \n\n numpy_output: {np_output}')


"""
output:

torch output: tensor([[[[1.1876]]]]) 

numpy_output: [[[[1.18756902]]]]
"""
