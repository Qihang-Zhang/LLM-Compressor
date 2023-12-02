import pdb
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from copy import copy

class Token_Buffer:
    def __init__(self):
        self.buffer = []
    
    def update(self, token_list):
        for token in token_list:
            self.buffer.append(token)
            
    def is_empty(self):
        if len(self.buffer) == 0:
            return True
        else:
            return False
        
    def pop(self):
        if self.is_empty():
            return None
        else:
            return self.buffer.pop(0)
            
    def display(self):
        print("buffer: ", "length", len(self.buffer))
        
class Input_Buffer:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.buffer = []
        begin_token = self.tokenizer(self.tokenizer.bos_token)["input_ids"][0]
        self.buffer.append(begin_token)
        
        
    def update(self, token):
        self.buffer.append(token)
        self.check_length()
    
    def check_length(self):
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)
            self.check_length()
        else:
            return
        
    def get_tensor(self):
        input = self.tokenizer(self.tokenizer.bos_token)
        input["input_ids"] = torch.tensor(self.buffer).unsqueeze(0)
        input["attention_mask"] = torch.ones_like(input["input_ids"])
        return input
    
    def display(self):
        print("tensor.input_ids: ", "shape", self.get_tensor()["input_ids"].shape, "value", self.get_tensor()["input_ids"])
        print("tensor.attention_mask: ", "shape", self.get_tensor()["attention_mask"].shape, "value", self.get_tensor()["attention_mask"])
    
class average:
    def __init__(self):
        self.sum = 0
        self.count = 0
        
    def update(self, value):
        self.sum += value
        self.count += 1
        
    def get(self):
        return self.sum / self.count
    
    def display(self):
        print("average: ", "value", self.get())
        
class average_ratio():
    def __init__(self):
        self.after = 0
        self.before = 0
        
    def update(self, after, before):
        self.after += after
        self.before += before
        
    def get(self):
        return self.after / self.before
    
    def display(self):
        print("average: ", "ratio", self.get())
        
if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    Input_Buffer = Input_Buffer(tokenizer)
    Token_Buffer = Token_Buffer()
    Token_Buffer.display()
    for i in range(1032):
        Token_Buffer.update([i])
        Token_Buffer.display()
    
    for i in range(1035):
        token = Token_Buffer.pop()
        if token is not None:
            print(token)
            Token_Buffer.display()
            Input_Buffer.update(token)
            Input_Buffer.display()
        else:
            print("None")
    pass
    


