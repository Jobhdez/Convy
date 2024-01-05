import torch
from src.frontend.torchfx import Conv2dNode
import torch.fx

def to_c(node):
    print(type(node))
    if isinstance(node, Conv2dNode):
            
        c_str = ""
        c_str = c_str = "#include runtime.c\n"

        bias = str(node.bias)
        input_h = str(node.input_height)
        input_w = str(node.input_width)
        filter_h = str(node.filter_height)
        batch_s = str(node.batch_size)
        channels = str(node.channels)

        tensor = torch_tensor_to_c(node.input_tensor)
        weight = torch_tensor_to_c(node.weight)

        c_str = c_str + "int main() {\n\n"
        c_str = c_str + f'int batch_size = {str(int(batch_s))};\n\n'
        c_str = c_str + f'int channels = {str(int(channels))}\n\n;'
        c_str = c_str + f'int input_height = {str(int(input_height))};\n\n'
        c_str = c_str + f'int input_width = {str(int(input_width))}\n\n;'
        c_str = c_str + f'int filter_height = {str(int(filter_height))}\n\n;'
        c_str = c_str + f'int filter_width = {str(int(filter_width))}\n\n;'

        c_str = c_str + f'float input_data[{batch_s}][{channels}][{input_h}][{input_w}] = {tensor};\n\n'
        c_str = c_str + f'float weight[{batch_s}][{channels}][{filter_h}][{filter_w}] = {weight};\n\n'
        c_str = c_str + f'float bias = {bias};'
        c_str = c_str + f'float output[{batch_size}][{channels}][{filter_h}][{filter_h}];\n\n'
        c_print = """ convolution(input_data, weight, &bias, batch_size, channels, input_height, input_width, filter_height, filter_width, output);\n\n

    // Print the result
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < 1; ++j) {  // Assuming output has only one channel
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    printf("%f ", output[i][j][k][l]);
                }
                printf("\n");
            }
        }
    }"""
    c_str = c_str + c_print + 'return 0;\n}'
    return c_str


                
def torch_tensor_to_c(tensor):
    c_array = str(tensor.numpy())
    c_array = c_array.replace('[', '{{').replace(']', '}}').replace('array', '').replace('(', '').replace(')', '')
    c_array = c_array.replace('\n', '').replace(' ', ',')
    return c_array
