from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import ollama

        
class Model:
    def __init__(self, model, gpu_num=0):
    
        self.ctx_len = 8192
        self.model_id = model
        self.__name__ = str([c for c in model if c.isalnum()])
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = f"cuda:{gpu_num}"
        self.model = self.model.to(self.device)
    
    def num_tokens(self, text):
        return len(self.tokenizer(text)["input_ids"])
    
    def cut_text(self, text, num_tokens):
        tokens = self.tokenizer(text)["input_ids"]
        tokens = tokens[:self.ctx_len - num_tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def generate(self, prompt: str)-> str:
        # tokenize, move to gpu
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        # check context lenght is not exceeded
        prompt_length = inputs['input_ids'].shape[1]
        if prompt_length > self.ctx_len:
            print(f"Context lenght exceeded! {prompt_length}")

        # generate
        response = self.model.generate(
            **inputs,
            max_length= 10000,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        decoded_response = self.tokenizer.decode(response[0][prompt_length:], skip_special_tokens=True)
        return decoded_response
    
    def to(self, device):
        self.device = device
        self.model.to(device)
