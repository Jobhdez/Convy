import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)

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
