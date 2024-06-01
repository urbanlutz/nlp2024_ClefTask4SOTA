from datetime import datetime
from typing import Callable
from tqdm import tqdm
from src.dataset import TDMSDataset, PATH, LogResult
from src.models import Model

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
            name: str
            ):
        self.model = model
        self.extract = extract
        self.name = name
        self.prompt_template = prompt_template
    
    def __call__(self, sample: str) -> str:
        return self.extract(self.model, self.prompt_template, sample)


def run(
        experiment: Experiment,
        data_path: PATH,
        max_iter = None
        ):
    """
    extract_fn: A function passing the text to a language model, augmented by a prompt, to extract TDMS quadruples
    run_name: Information about the run, model name, experiment setup etc
    """
    # define run name
    now_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_id = f"{PATH.get_name(data_path)}_{experiment.name}-{now_str}"


    dataset = TDMSDataset(data_path)
    if max_iter:
        dataset.all_paths = dataset.all_paths[:max_iter]
    logger = LogResult(run_id, do_write=True, additional_col_names=["ground_truth"])

    indexes = len(dataset)
    for i in tqdm(range(indexes)):
        f, tex, ground_truth = dataset[i]
        model_output = experiment(tex)

        logger.log(f, str(model_output), str(ground_truth))
    df = logger.save()
    return df


# def run(
#         extract_fn: Callable[[str], str],
#         data_path: PATH,
#         run_name:str,
#         max_iter = None
#         ):
#     """
#     extract_fn: A function passing the text to a language model, augmented by a prompt, to extract TDMS quadruples
#     run_name: Information about the run, model name, experiment setup etc
#     """
#     # define run name
#     now_str = datetime.now().strftime('%Y%m%d-%H%M%S')
#     run_id = f"{PATH.get_name(data_path)}_{run_name}-{now_str}"


#     dataset = TDMSDataset(data_path)
#     if max_iter:
#         dataset.all_paths = dataset.all_paths[:max_iter]
#     logger = LogResult(run_id, do_write=True, additional_col_names=["ground_truth"])

#     indexes = len(dataset)
#     for i in tqdm(range(indexes)):
#         f, tex, ground_truth = dataset[i]
#         model_output = extract_fn(tex)

#         logger.log(f, str(model_output), str(ground_truth))
#     df = logger.save()
#     return df
