import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    def __init__(self, model):
        super().__init__()
        self.model_id = model
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    def generate(self, prompt: str)-> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        response = self.model.generate(**inputs, max_length= 20)
        return self.tokenizer.decode(response[0], skip_special_tokens=True)
    