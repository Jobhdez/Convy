import torch
#from src.frontend.torchfx import Conv2dNode
import torch.fx
from src.frontend.nodes import Conv2dNode
            
def to_c(node):
    print(f'hello: {type(node)}')
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

