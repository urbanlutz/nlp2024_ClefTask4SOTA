from datetime import datetime
from typing import Callable
from tqdm import tqdm
from src.dataset import TDMSDataset, PATH, LogResult
from src.models import Model
import torch
import gc
import time
def free_cuda_memory():
    torch.cuda.empty_cache()
    gc.collect()

class Experiment:
    """
    model: A Model from src.models
    prompt_template: A function that inserts the tex segment into a prompt
    extract: a function taking a model, a prompt template and a tex file, defining how the model with the prompt tempalte gets applied to the tex file
    """
    def __init__(
            self,
            model: Model,
            prompt_template: Callable[[str], str],
            extract: Callable[[Model, Callable[[str], str], str], str],
            ):
        self.model = model
        self.extract = extract
        self.name = f"{extract.__name__}-{prompt_template.__name__}-{model.__name__}"
        self.prompt_template = prompt_template
    
    def __call__(self, sample: str) -> str:
        extracted_text = self.extract(sample)
        truncated = self.model.cut_text(extracted_text, self.model.num_tokens(self.prompt_template("")))
        prompt = self.prompt_template(truncated)
        response = self.model.generate(prompt)
        return prompt, response
    
    def __del__(self):
        del self.model

    def __repr__(self):
        return self.name


def run(
        experiment: Experiment,
        data_path: PATH,
        max_iter = None,
        post_proc = None
        ):
    # define run name
    now_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_id = f"{PATH.get_name(data_path)}_{experiment.name}-{now_str}"


    dataset = TDMSDataset(data_path)
    if max_iter:
        dataset.all_paths = dataset.all_paths[:max_iter]
        
    print(run_id)
    logger = LogResult(run_id, save_interval=1, do_write=True, additional_col_names=["prompt", "raw", "ground_truth", "inference_s"])

    if not experiment.model.is_loaded:
        experiment.model.load()
    indexes = len(dataset)
    for i in tqdm(range(indexes)):
        f, tex, ground_truth = dataset[i]
        t_start = time.perf_counter()
        prompt, model_output = experiment(tex)
        t_end = time.perf_counter()
        elapsed_time = t_end- t_start
        if post_proc:
            processed = post_proc(model_output)
        logger.log(f, str(processed), str(prompt), str(model_output), str(ground_truth), elapsed_time)
    df = logger.save()
    del experiment
    free_cuda_memory()
    return df