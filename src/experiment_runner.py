
import tqdm
from tqdm import tqdm
from src.dataset import TDMSDataset, PATH,LogResult
from datetime import datetime


from typing import Callable


def run(
        extract_fn: Callable[[str], str], 
        data_path: PATH, 
        run_name:str
        ):
    """
    extract_fn: A function passing the text to a language model, augmented by a prompt, to extract TDMS quadruples
    run_name: Information about the run, model name, experiment setup etc
    """
    # define run name
    now_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_id = f"{PATH.get_name(data_path)}_{run_name}-{now_str}"


    dataset = TDMSDataset(data_path)
    logger = LogResult(run_id, do_write=True, additional_col_names=["ground_truth"])

    indexes = len(dataset)
    for i in tqdm(range(indexes)):
        f, tex, ground_truth = dataset.__getitem__(i)
        model_output = extract_fn(tex)

        logger.log(f, str(model_output), str(ground_truth))
    df = logger.save()
    return df
