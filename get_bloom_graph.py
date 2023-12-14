from transformers import AutoTokenizer, BloomModel, BloomConfig
import torch

config = BloomConfig()
config.torchscript = True

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = BloomModel(config)
model.eval()
model = BloomModel.from_pretrained("bigscience/bloom-560m", torchscript=True)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
t1 = inputs['input_ids']
t2 = inputs['attention_mask']


module = torch.jit.trace(model, t1)
state_dict = module.state_dict()
graph = module.graph.copy()
