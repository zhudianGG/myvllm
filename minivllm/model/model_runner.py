# model/model_runner.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelRunner:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt")
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def forward(self, input_ids, past_key_values=None):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
        return outputs