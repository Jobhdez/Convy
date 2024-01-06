import torch
import torch.nn as nn
import numpy as np
import numpy
import torch.nn.functional as F
from src.frontend.extract_tensor_data import ShapeProp
from src.backend.lalg_to_c import to_c
from src.frontend.torch_to_ast import torch_to_ast
from src.backend.nn_operators import convolution_torch

# == utils for the example ==

def write_file(c_program, file_name):
    with open(file_name, "w") as f:
        f.write(c_program)

def test_my_conv_fn(input_tensor, net):
    module = torch.jit.trace(net, input_tensor)
    state_dict = module.state_dict()
    weight = state_dict['conv.weight']
    bias = state_dict['conv.bias']
    saved_module = module.save("test_conv.pth")
    load_module = torch.jit.load("test_conv.pth")

    with torch.no_grad():
        torch_output = load_module(input_tensor)

    my_conv_output = convolution_torch(input_tensor, weight, bias)

    print(f'torch_output: {torch_output}\n my_conv_output: {my_conv_output}')

# == Example code for the compilation of a conv layer to C ===

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3).float()

    def forward(self, x):
        return self.conv(x)

net = Net()
ones = np.ones((1,1,3,3), dtype=np.float32)
input_tensor = torch.tensor(ones, dtype=torch.float32)

nn_ast = torch_to_ast(net, input_tensor)
c_program = to_c(nn_ast[0])

print("\ntesting my conv fn....")
test_my_conv_fn(input_tensor, net)

print("\n\nwriting generated C file...")
write_file(c_program, "../backend/conv.c")
