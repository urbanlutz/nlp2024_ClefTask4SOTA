from torch.utils.data import Dataset

import os
import pandas as pd
class PATH:
    TRAIN = "./data/train"
    VAL = "./data/validation"
    TEST = "./data/test2-zero-shot-papers"

    def get_name(path):
        return {v:k for k, v in PATH.__dict__.items()}[path]

UNANSWERABLE = "unanswerable\n"


class LogResult:
    def __init__(self, run, save_interval=10, do_write = True, additional_col_names=[]):
        self.run = run
        self.results = []
        self.save_interval = save_interval
        self.do_write = do_write
        self.additional_col_names = additional_col_names

    def log(self, f, annotation, *args):
        
        self.results.append((self.run, f, annotation, *args))
        if self.do_write:
            self._write_annotation_file(f, annotation)
            if len(self.results) % self.save_interval == 0:
                _ = self._write_feather()

    def _write_annotation_file(self, f, annotation):
        filename = f"results/{self.run}/{f}/annotations.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(str(annotation))

    def _write_feather(self):
        standard_col_names = ["run", "f", "annotation"]
        df = pd.DataFrame(self.results)
        df = df.rename({i: c for i, c in enumerate([*standard_col_names, *self.additional_col_names])}, axis=1)
        if self.do_write:
            df.to_feather(f"results/{self.run}/df.feather")
        return df

    def save(self):
        return self._write_feather()


def path_join(*args):
    return os.path.join(*args).replace('\\', '/') 

def find(extension, *args):
    p = path_join(*args)
    res = list(filter(lambda x: x.endswith(extension), os.listdir(p)))
    if res:
        return path_join(p, res[0])
    else:
        return None

def _read_files(i, tex_path, jsn_path):
    tex, jsn = None, None

    try:
        with open(tex_path) as f:
            tex = f.read()
    except: # tex not read
        pass
    
    if jsn_path:
        with open(jsn_path, encoding="utf-8") as f:
            jsn = f.read()
            try:
                jsn = eval(jsn)
            except:
                pass #"unanswerable" is not a dict, no eval possible/necessary
    
    return i, tex, jsn

class TDMSDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_paths = [(p, find("tex", path, p), find("json", path, p)) for p in os.listdir(path)]
    
    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        i, tex_path, jsn_path = self.all_paths[idx]
        return _read_files(i, tex_path, jsn_path)
    
class BinaryTDMSDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_paths = [(p, find("tex", path, p), find("json", path, p)) for p in os.listdir(path)]
    
    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        i, tex_path, jsn_path = self.all_paths[idx]
        i, t, j =  _read_files(i, tex_path, jsn_path)
        has_tdms = j != UNANSWERABLE if j is not None else None
        return i, t, has_tdms
    
    def get_dataloader(self):
        pass

