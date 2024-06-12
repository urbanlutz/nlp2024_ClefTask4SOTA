from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import namedtuple
import torch
import ollama

ModelId = namedtuple("ModelId", "hf ollama")

class ARCHITECTURE:
    MISTRAL_7b = ModelId("mistralai/Mistral-7B-v0.3","mistral:7b")
    MISTRAL_7b_IT = ModelId("mistralai/Mistral-7B-Instruct-v0.3", "mistral:instruct")
    GEMMA_7b = ModelId("google/gemma-7b", "gemma:7b")
    GEMMA_2b = ModelId("google/gemma-2b", "gemma:2b")
    # GEMMA_IT = ModelId("google/gemma-7b-it", "None")
    LLAMA_8b = ModelId("meta-llama/Meta-Llama-3-8B", "llama3:8b")
    LLAMA_8b_IT = ModelId("meta-llama/Meta-Llama-3-8B-Instruct", "llama3:instruct")
    LLAMA_70b = ModelId("meta-llama/Meta-Llama-3-70B", "llama3:70b")
    MIXTRAL_22B = ModelId("mistralai/Mixtral-8x22B-v0.1", "mixtral:8x22b")
    MIXTRAL_7B = ModelId("mistralai/Mixtral-8x7B-v0.1", "mixtral:8x7b")


class OllamaModel:
    def __init__(self, model:ModelId):
        self.ctx_len = 8192
        self.model_id = model
        self.__name__ = ''.join([c for c in model.ollama if c.isalnum()])
        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(model.hf)
        self.is_loaded = True

    def load(self): # ollama models are loaded on demand by ollama
        pass

    def generate(self, prompt: str) -> str:
        prompt_length = len(self.tokenizer(prompt)["input_ids"])
        if prompt_length >= self.ctx_len:
            print(f"Context lenght exceeded! {prompt_length}")
        res = None
        try:
            res = ollama.generate(model=self.model_id.ollama, prompt=prompt, options={"temperature": 0, "num_ctx":self.ctx_len})
            res = res["response"]
            return res
        except Exception as ex:
            print(ex)
            return f"ollama error: {ex}"
        
    def num_tokens(self, text):
        return len(self.tokenizer(text)["input_ids"])
    
    def cut_text(self, text, num_tokens):
        tokens = self.tokenizer(text)["input_ids"]
        tokens = tokens[:self.ctx_len - num_tokens -1]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
        
class Model:
    def __init__(self, model:ModelId, gpu_num=0):
        self.ctx_len = 8192
        self.model = model
        self.__name__ = ''.join([c for c in model.hf if c.isalnum()])
        self.device = "cuda"
        self.is_loaded = False
    
    def load(self):
        if not self.is_loaded:
            self.model = AutoModelForCausalLM.from_pretrained(self.model.hf, torch_dtype=torch.bfloat16, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.hf)

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
        if prompt_length >= self.ctx_len:
            print(f"Context lenght exceeded! {prompt_length}")

        # generate
        response = self.model.generate(
            **inputs,
            max_length= 10000,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        # inputs = inputs.to("cpu")
        # del inputs
        decoded_response = self.tokenizer.decode(response[0][prompt_length:], skip_special_tokens=True)
        return decoded_response
    
    # def to(self, device):
    #     self.device = device
        # self.model.to(device)