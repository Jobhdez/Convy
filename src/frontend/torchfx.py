import torch
import torch.fx
import traceback
import numpy as np
from torch._dispatch.python import enable_python_dispatcher
from torch.fx.node import Node, map_aggregate
from typing import Any, Tuple, NamedTuple, Optional, Dict
from torch.fx._compatibility import compatibility
from torch._guards import detect_fake_mode
### this is taken from the pytorch docs
__all__ = ['TensorMetadata', 'ShapeProp']

@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape : torch.Size
    dtype : torch.dtype
    requires_grad : bool
    stride : Tuple[int, ...]
    memory_format : Optional[torch.memory_format]
    data: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor 

    # Quantization metadata
    is_quantized : bool
    qparams: Dict[str, Any]
    

def _extract_tensor_metadata(result : torch.Tensor, include_contiguity=True) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()

    memory_format = None

    if include_contiguity:
        memory_formats = {
            torch.contiguous_format,
            torch.channels_last,
            torch.channels_last_3d,
        }
        for query_format in memory_formats:
            if result.is_contiguous(memory_format=query_format):
                memory_format = query_format
                break

    is_quantized = result.is_quantized
    qparams: Dict[str, Any] = {}
    if is_quantized:
        qscheme = result.qscheme()
        qparams["qscheme"] = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            qparams["scale"] = result.q_scale()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_zero_point()  # type: ignore[assignment]
        elif qscheme in {torch.per_channel_affine, torch.per_channel_affine_float_qparams, torch.per_channel_symmetric}:
            # In this branch, scale and zero_point are expected to be tensors,
            # we store the values as immutable_list in TensorMetadata for
            # easier serialization downstream
            qparams["scale"] = result.q_per_channel_scales().tolist()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_per_channel_zero_points().tolist()  # type: ignore[assignment]
            qparams["axis"] = result.q_per_channel_axis()  # type: ignore[assignment]
    result_data = result.detach()

    weight = None
    bias = None

    if result.requires_grad and len(result.shape) == 4:  # Assuming ConvNet weights are 4-dimensional
        if result.grad_fn is not None and "Conv" in str(result.grad_fn):
            weight = result.clone().detach().requires_grad_(False)
            bias = result.grad_fn.next_functions[1][0].variable.clone().detach().requires_grad_(False)


    return TensorMetadata(
        shape, dtype, requires_grad, stride, memory_format, result_data,weight, bias, is_quantized, qparams)

@compatibility(is_backward_compatible=True)
class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and
    record the shape and type of the result
    into the corresponding node.

    Example:
         In this example, we record the shape
         and data type of a module given
         an example input ``torch.randn(50, D_in)``.
         We print the name, shape and dtype of each node.

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super().__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)
            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred
        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        model = TwoLayerNet(D_in, H, D_out)
        gm = torch.fx.symbolic_trace(model)
        sample_input = torch.randn(50, D_in)
        ShapeProp(gm).propagate(sample_input)

        for node in gm.graph.nodes:
            print(node.name, node.meta['tensor_meta'].dtype,
                node.meta['tensor_meta'].shape)

        The output of this code is:

        x torch.float32 torch.Size([50, 1000])
        linear1 torch.float32 torch.Size([50, 100])
        clamp_1 torch.float32 torch.Size([50, 100])
        linear2 torch.float32 torch.Size([50, 10])
        output torch.float32 torch.Size([50, 10])

    Args:
         module (GraphModule): The module to be executed
         fake_mode (FakeTensorMode): A fake mode for copying the gm

    """
    def __init__(self, gm, fake_mode=None):
        super().__init__(gm)
        if fake_mode is None:
            fake_mode = detect_fake_mode()
        if fake_mode is not None:
            from torch._dynamo.utils import deepcopy_to_fake_tensor
            # Note:
            # We need fake execution cause the inputs are fake, however, we cannot fakify the module
            # - because we need to write to the tensor_meta of the real module. So we fakify to
            # produce a result (L131 below), to extract tensor meta, and then keep going.
            #
            # If we were to fakify, we would write to the wrong node, and then downstream fusion
            # would be missing the tensor_meta.
            #
            # See torch/_inductor/overrides.py for where this is called upstream of fusion.
            self.fake_module = deepcopy_to_fake_tensor(self.module, fake_mode)
            self.fake_mode = fake_mode
        else:
            self.fake_module = None
            self.fake_mode = None

        self.real_module = self.module

    def run_node(self, n : Node) -> Any:
        try:
            if self.fake_module is not None:
                # Hacky swap. Alternatively, we could do this with overriding
                # call_module and get_attr.
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    with self.fake_mode, enable_python_dispatcher():
                        result = super().run_node(n)
                else:
                    result = super().run_node(n)
            finally:
                self.module = self.real_module
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"ShapeProp error for: node={n.format_node()} with "
                f"meta={n.meta}"
            ) from e

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['tensor_meta'] = meta
            

        n.meta['type'] = type(result)
        return result

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        if self.fake_mode is not None:
             fake_args = [self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args]
            #fake_args = [t.detach() if isinstance(t, torch.Tensor) else t for t in args]
        else:
            fake_args = args
        return super().run(*fake_args)
        #return fake_args

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3).float()

    def forward(self, x):
        return self.conv(x)

""" 
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
model = TwoLayerNet(D_in, H, D_out)
gm = torch.fx.symbolic_trace(model)
sample_input = torch.randn(50, D_in)
ShapeProp(gm).propagate(sample_input)
for node in gm.graph.nodes:
    print(node.name, node.meta['tensor_meta'].dtype,
          node.meta['tensor_meta'].shape, node.meta['tensor_meta'].data)

for node in gm.graph.nodes:
    type(node)

gm.graph.print_tabular()
"""
def get_layers(graph):
    
    return list(graph.named_modules())

"""
net = Net()
gm2 = torch.fx.symbolic_trace(net)
sample = torch.rand(1, 1, 3, 3)
module = torch.jit.trace(net, sample)
state_dict = module.state_dict()
"""

"""
print("-----print sample----")
print(sample)
shape_prop = ShapeProp(gm2)
shape_prop.propagate(sample)
print("\n")
print("---meta-data")
for node in gm2.graph.nodes:
    print(node.name, node.meta['tensor_meta'].dtype,
          node.meta['tensor_meta'].shape, node.meta['tensor_meta'].data,
          node.meta['tensor_meta'].weight, node.meta['tensor_meta'].bias)

print(get_layers(gm2))

print("\nstate_dict")
weight = state_dict['conv.weight']
bias = state_dict['conv.bias']

print(state_dict)
"""
def convolution_torch(input_data, weight, bias):
    # Assuming 'input_data' is a 4D tensor (batch_size, channels, height, width)
    _, _, input_height, input_width = input_data.shape
    _, _, filter_height, filter_width = weight.shape 

    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1

    output = torch.zeros(1, 1, output_height, output_width)

    for h in range(output_height):
        for w in range(output_width):
            receptive_field = input_data[:, :, h:h+filter_height, w:w+filter_width]
            output[:, :, h, w] = torch.sum(receptive_field * weight) + bias

    return output
"""
print("\n convolution result")
result = convolution_torch(sample, weight, bias)
print(result)

layers = get_layers(gm2)
"""
class Conv2dNode:
    def __init__(self, input_tensor, weight, bias, input_height, input_width, filter_height, filter_width, batch_size, channels):
        self.input_tensor = input_tensor
        self.weight = weight
        self.bias = bias
        self.input_height = input_height
        self.input_width = input_width
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.batch_size = batch_size
        self.channels = channels

net = Net()
ones = np.ones((1,1,3,3), dtype=np.float32)
input_tensor = torch.tensor(ones, dtype=torch.float32)
def torch_to_ast(net, input_tensor):
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
    ast_nodes = []
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
            weight = state_dict[input_name + ".weight"]
            bias_tensor = state_dict[bias_name]
            batch_size, channels, input_height, input_width = input_tensor.shape
            _,_, filter_height, filter_width = weight.shape
            bias = None
            if len(bias_tensor) == 1 and isinstance(float(bias_tensor[0]), float):
                bias = float(bias_tensor[0])
            else:
                bias = bias_tensor
            ast_nodes.append(Conv2dNode(input_tensor, weight, bias, input_height, input_width, filter_height, filter_width, batch_size, channels))
    return ast_nodes
        


conv2d = torch_to_ast(net, input_tensor)

### test


ones = np.ones((1, 1, 3, 3), dtype=np.float32)

ones_pytorch = torch.tensor(ones, dtype=torch.float32)
module = torch.jit.trace(net, ones_pytorch)
state_dict = module.state_dict()
weight = state_dict['conv.weight']
bias = state_dict['conv.bias']

savm = module.save('test1.pth')
loadm = torch.jit.load('test1.pth')

with torch.no_grad():
    torch_output = loadm(ones_pytorch)

np_output = convolution_torch(ones_pytorch, weight, bias)

print(f'torch output: {torch_output} \n\n numpy_output: {np_output}')

def to_c(node):
    print(type(node))
    if isinstance(node, Conv2dNode):
            
        c_str = ""
        c_str = c_str = '#include "runtime.c"\n'
        c_str = c_str + "#include <stdio.h>\n\n"

        bias = str(node.bias)
        input_h = str(node.input_height)
        input_w = str(node.input_width)
        filter_h = str(node.filter_height)
        filter_w = str(node.filter_width)
        batch_s = str(node.batch_size)
        channels = str(node.channels)

        tensor = torch_tensor_to_c(node.input_tensor)
        weight = torch_tensor_to_c(node.weight)
        
        c_str = c_str + "\nint main() {\n\n"
        c_str = c_str + f'int batch_size = {str(int(batch_s))};\n\n'
        c_str = c_str + f'int channels = {str(int(channels))};\n\n'
        c_str = c_str + f'int input_height = {str(int(input_h))};\n\n'
        c_str = c_str + f'int input_width = {str(int(input_w))};\n\n'
        c_str = c_str + f'int filter_height = {str(int(filter_h))};\n\n'
        c_str = c_str + f'int filter_width = {str(int(filter_w))};\n\n'

        c_str = c_str + f'float input_data[{batch_s}][{channels}][{input_h}][{input_w}] = {tensor};\n\n'
        c_str = c_str + f'float weight[{batch_s}][{channels}][{filter_h}][{filter_w}] = {weight};\n\n'
        c_str = c_str + f'float bias = {bias};\n'
        c_str = c_str + f'float output[{batch_s}][{channels}][{filter_h}][{filter_h}];\n\n'
        c_print = """ convolution(input_data, weight, &bias, batch_size, channels, input_height, input_width, filter_height, filter_width, output);\n\n

    // Print the result
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < 1; ++j) {  // Assuming output has only one channel
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    printf("%f ", output[i][j][k][l]);
                }
                
            }
        }
    }"""
    c_str = c_str + c_print + 'return 0;\n}'
    return c_str


                
def torch_tensor_to_c(tensor):
    
    c_array = tensor.numpy()
    c_array = c_array.tolist()
    c_array = str(c_array)
    print(c_array)
    c_array = c_array.replace('[', '{').replace(']', '}')
    return c_array


c_str = to_c(conv2d[0])

with open("conv.c", "w") as f:
    f.write(c_str)
    

