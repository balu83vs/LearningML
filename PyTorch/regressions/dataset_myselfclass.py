import torch
from torch.utils.data import Dataset

from PIL import Image

import os
import json
import numpy as np




class DatasetReg(Dataset):

    def __init__(self, path, transform = None):
        self.path = path
        self.transform = transform

        self.list_name_file = os.listdir(path)

        if 'coords.json' in self.list_name_file:
            self.list_name_file.remove('coords.json')

        self.len_dataset = len(self.list_name_file)

        with open(os.path.join(self.path, 'coords.json'), 'r') as f:
            self.dict_coords = json.load(f)    

    
    def __len__(self):
        return self.len_dataset


    def __getitem__(self, index):
        name_file = self.list_name_file[index]
        path_img = os.path.join(self.path, name_file)

        if self.transform is not None:
            img = Image.open(path_img)
            coord = torch.tensor(self.dict_coords[name_file], dtype=torch.float32)
            img = self.transform(img)
        else:
            img = np.array(Image.open(path_img))
            coord = np.array(self.dict_coords[name_file])    

        return img, coord    
