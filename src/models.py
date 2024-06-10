from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import ollama

class Model:
    def generate(self, prompt: str)-> str:
        pass

class OllamaModel(Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def generate(self, prompt: str) -> str:
        res = None
        try:
            res = ollama.generate(model=self.model, prompt=prompt, options={"temperature": 0})
            res = res["response"]
            return res
        except Exception as ex:
            print(ex)
            return res
        
class HFModel(Model):
    def __init__(self, model, gpu_num=0):
        super().__init__()
        self.model_id = model
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = f"cuda:{gpu_num}"
        self.model = self.model.to(self.device)
    
    def num_tokens(self, input):
        return len(self.tokenizer(input)["input_ids"])
    
    def cut_text(self, text, num_tokens):
        tokens = self.tokenizer(text)["input_ids"]
        tokens = tokens[:num_tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def generate(self, prompt: str)-> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        prompt_length = inputs['input_ids'].shape[1]
        response = self.model.generate(**inputs, max_length= 10000, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(response[0][prompt_length:], skip_special_tokens=True)
    