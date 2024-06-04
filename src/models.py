from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama


def _convert_tdms_to_tuple(model_output_parsed):
    tuples = []
    for item in model_output_parsed:
        try:
            t = ((item["Task"], item["Dataset"],item["Metric"],item["Score"]))
            tuples.append(t)
        except:
            # parse error, ignore instance
            pass
    return tuples

def _format_tdms(tuples):
    """make unique, format as string"""
    unique = set(tuples)
    dicts = [{"LEADERBOARD": {
        "Task": t,
        "Dataset":d,
        "Metric":m,
        "Score":s
    }} for t,d,m,s in unique]
    return str(dicts)

class Model:
    def generate(self, prompt: str)-> str:
        pass

    def parse_response(self, response):
        try:
            response = "[" + res.split("[", 1)[-1].rsplit("]", 1)[0] + "]"
            response = json.loads(response)
            response = _convert_tdms_to_tuple(response)
            return _format_tdms(response)
        except:
            return str(response)

class OllamaModel(Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def generate(self, prompt: str) -> str:
        res = None
        try:
            res = ollama.generate(model=self.model, prompt=prompt, options={"temperature": 0})
            res = res["response"]
            res = self.parse_response(res)
            return res
        except Exception as ex:
            print(ex)
            return f"ollama error: {ex}"
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
    