from torch.utils.data import Dataset, DataLoader


import glob
import os
import json

TRAIN_PATH = "./data/train"

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
        self.all_paths = [(p, find("tex", TRAIN_PATH, p), find("json", TRAIN_PATH, p)) for p in os.listdir(TRAIN_PATH)]
    
    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        return self._read(idx)

    def _read(self, idx):
        tex, jsn = None, None
        try:
            i, tex_path, jsn_path = all_paths[idx]
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
    
    def get_trainloader(self):
        return DataLoader(transformed_dataset, batch_size=32, shuffle=True)