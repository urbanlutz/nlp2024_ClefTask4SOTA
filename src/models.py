from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama

class Model:
    def generate(self, prompt: str)-> str:
        pass

class LLama(Model):
    def __init__(self, model):
        self.model = model
    
    def generate(self, prompt: str) -> str:
        try:
            res = ollama.generate(model=self.model, prompt=prompt, options={"temperature": 0})
            return res["response"]
        except Exception as ex:
            print(ex)
            return f"ollama error: {ex}"
        
class HFModel(Model):
    def __init__(self, model):
        self.model_id = model
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    def generate(self, prompt: str)-> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        response = self.model.generate(**inputs, max_length= 20)
        return self.tokenizer.decode(response[0], skip_special_tokens=True)
    