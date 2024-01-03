import torch
import torch.nn as nn
import numpy as np
import numpy
import torch.nn.functional as F

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
#### word embdeeings
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
# Print the first 3, just so you can see what they look like.
print(ngrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

vocab_size = 10000
embedding_dim = 50
context_size = 2

# Create a random input tensor with the shape (batch_size, context_size)
batch_size = 1  # You can change this based on your actual use case
input_tensor = torch.randint(low=0, high=vocab_size, size=(batch_size, context_size))

# Instantiate the model
model2 = NGramLanguageModeler(vocab_size, embedding_dim, context_size)

# Trace the model with the example input tensor
traced_model = torch.jit.trace(model2, input_tensor)

"""
>>> from compiler import get_node_inputs
>>> inpsf = get_node_inputs(traced_model)
>>> inpsf
['__torch__.conv2d.NGramLanguageModeler', '__torch__.conv2d.NGramLanguageModeler', '__torch__.conv2d.NGramLanguageModeler', '__torch__.torch.nn.modules.sparse.Embedding', 'Long(1, 2, strides=[2, 1], requires_grad=0, device=cpu)', 'int', 'int', 'Tensor', 'int[]', '__torch__.torch.nn.modules.linear.Linear', 'Float(1, 100, strides=[100, 1], requires_grad=1, device=cpu)', 'Tensor', '__torch__.torch.nn.modules.linear.___torch_mangle_2.Linear', 'Float(1, 128, strides=[128, 1], requires_grad=1, device=cpu)', 'Tensor', 'int', 'NoneType']
"""
