import os
from transformers import AutoModel
model_path = "/home/dataset-local/AntigenLM_V2/LLM/esm2_650M"
model = AutoModel.from_pretrained(model_path)
for name, param in model.named_parameters():
    print(name)
