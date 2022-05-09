from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import csv
import cv2 as cv
import time
import numpy as np
import h5py
import csv
import glob

res = glob.glob("../data/AZ9010_256/*.png")

def default_loader(path):
    img = io.imread(path)
    y = np.round(img/16.0)
    y = torch.from_numpy(y[0])
    y = y.type(torch.LongTensor)

    img = img/255.0
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, axis=0)
    img = img.type(torch.FloatTensor)

    return img, y 


class Radars(Dataset):
    def __init__(self,transform=None):
        super(Radars, self).__init__()
        self.radar = res

    def __getitem__(self, index):
        img, y = default_loader(self.radar[index])

        return img, y


    def __len__(self):
        return 3000
