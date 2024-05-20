from torch.utils.data import Dataset, DataLoader

import os
import json

class PATH:
    TRAIN = "./data/train"
    VAL = "./data/validation"
    TEST = "./data/test2-zero-shot-papers"

UNANSWERABLE = "unanswerable\n"


def write_annotation_file(run, f, annotation):
    filename = f"results/{run}/{f}/annotations.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(str(annotation))

def path_join(*args):
    return os.path.join(*args).replace('\\', '/') 

def find(extension, *args):
    p = path_join(*args)
    res = list(filter(lambda x: x.endswith(extension), os.listdir(p)))
    if res:
        return path_join(p, res[0])
    else:
        return None

class TDMSDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_paths = [(p, find("tex", path, p), find("json", path, p)) for p in os.listdir(path)]
    
    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        return self._read(idx)

    def _read(self, idx):
        tex, jsn = None, None
        try:
            i, tex_path, jsn_path = self.all_paths[idx]
        except Exception as ex:
            print(i)
            raise ex
        try:
            with open(tex_path) as f:
                tex = f.read()

            try:
                with open(jsn_path) as f:
                    jsn = json.load(f)
                    print("loaded json")
            except:
                with open(jsn_path) as f:
                    jsn = f.read()
                    jsn = eval(jsn)
            return i, tex, jsn
        except:
            return i, tex, jsn
    
    
class BinaryTDMSDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_paths = [(p, find("tex", path, p), find("json", path, p)) for p in os.listdir(path)]
    
    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        return self._read(idx)

    def _read(self, idx):
        tex, jsn = None, None
        try:
            i, tex_path, jsn_path = self.all_paths[idx]
        except Exception as ex: # Index not found
            print(i)
            raise ex
        
        try:
            with open(tex_path) as f:
                tex = f.read()
        except: # tex not read
            pass
        
        if jsn_path:
            with open(jsn_path) as f:
                jsn = f.read()
                try:
                    jsn = eval(jsn)
                except:
                    pass #"unanswerable" is not a dict, no eval possible/necessary
        has_tdms = jsn != UNANSWERABLE if jsn is not None else None
        return i, tex, has_tdms
    
    def get_dataloader(self):
        pass

