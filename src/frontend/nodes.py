# == nodes ==

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
