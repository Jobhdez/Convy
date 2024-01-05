import torch
import torch.nn as nn
from conv2d import get_nodes, get_inputs
from torchfx import ShapeProp, get_layers

def torch_to_ast(net, input_tensor):
    """
    constructs an AST from a neural net and input tensor.
    """
    module = torch.jit.trace(net, input_tensor)
    gm = torch.fx.symbolic_trace(net)
    layers = get_layers(gm)
    nodes = ShapeProp(gm)
    nodes.propagate(input_tensor)

    tensor_data = []
    for node in gm.graph.nodes:
        tensor_data.append({'name': node.name, 'dtype': node.meta['tensor_meta'].dtype, 'shape': node.meta['tensor_meta'].shape, 'tensor': node.meta['tensor_meta'].data})

    layers_lst = [list(layer) for layer in layers]
    print(layers_lst)
    layers = layers_lst[1:]
    print(layers)
    layers = [{'name': layer[0], 'nn_obj': layer[1]} for layer in layers]
    print(layers)
    print(type(layers[0]['nn_obj']))

    for layer in range(len(layers)):
        nn_obj = layers[layer]['nn_obj']
        print(nn_obj)
        print(type(nn_obj))
        if isinstance(nn_obj, torch.nn.modules.conv.Conv2d):
            input_tensor = None
            weight_tensor = None
            for tensor in tensor_data:
                if tensor['name'] == 'x':
                    input_tensor = tensor['tensor']
                elif tensor['name'] == layers[layer]['name']:
                    input_name = tensor['name']
                    weight_tensor = tensor['tensor']

            state_dict = module.state_dict()
            bias_name = input_name + ".bias"
            bias_tensor = state_dict[bias_name]

            return Conv2d(input_tensor, weight, bias)
        else:
            break
        
class Conv2d:
    def __init__(self, input_tensor, weight, bias):
        self.input_tensor = input_tensor
        self.weight = weight
        self.bias = bias
