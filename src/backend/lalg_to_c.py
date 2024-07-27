import torch
import torch.fx
from src.frontend.nodes import Conv2dNode
            
def to_c(node):
    if isinstance(node, Conv2dNode):
        c_str = """
#include "runtime.c"
#include <stdio.h>

int main() {
"""
        bias = str(node.bias)
        input_h = str(node.input_height)
        input_w = str(node.input_width)
        filter_h = str(node.filter_height)
        filter_w = str(node.filter_width)
        batch_s = str(node.batch_size)
        channels = str(node.channels)

        tensor = torch_tensor_to_c(node.input_tensor)
        weight = torch_tensor_to_c(node.weight)
        
        c_str += f"    int batch_size = {str(int(batch_s))};\n"
        c_str += f"    int channels = {str(int(channels))};\n"
        c_str += f"    int input_height = {str(int(input_h))};\n"
        c_str += f"    int input_width = {str(int(input_w))};\n"
        c_str += f"    int filter_height = {str(int(filter_h))};\n"
        c_str += f"    int filter_width = {str(int(filter_w))};\n"
        
        c_str += f"    float input_data[{batch_s}][{channels}][{input_h}][{input_w}] = {tensor};\n"
        c_str += f"    float weight[{batch_s}][{channels}][{filter_h}][{filter_w}] = {weight};\n"
        c_str += f"    float bias = {bias};\n"
        c_str += f"    float output[{batch_s}][{channels}][{filter_h}][{filter_h}];\n"
        
        c_str += """
    convolution(input_data, weight, &bias, batch_size, channels, input_height, input_width, filter_height, filter_width, output);

    printf("%f", output[0][0][0][0]);
       
    return 0;
}
"""
        return c_str

def torch_tensor_to_c(tensor):
    c_array = tensor.numpy().tolist()
    c_array = str(c_array).replace('[', '{').replace(']', '}')
    return c_array
