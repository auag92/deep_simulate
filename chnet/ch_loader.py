import torch
import warnings
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CahnHillDataset(Dataset):

    def __init__(self, arr_x, arr_y, transform_x=None, transform_y=None):

        self.arr_x = arr_x
        self.arr_y = arr_y
        self.transform_x = transform_x
        self.transform_y = transform_y
        assert len(arr_x) ==len(arr_y)

    def __len__(self):
        return len(self.arr_x)

    def __getitem__(self, idx):

        im_x = self.arr_x[idx]
        im_y = self.arr_y[idx]

        if self.transform_x:
            im_x = self.transform_x(im_x)
        if self.transform_y:
            im_y = self.transform_y(im_y)
        sample = {'x': torch.tensor(im_x).double(), 'y': torch.tensor(im_y).double()}
        
        return sample