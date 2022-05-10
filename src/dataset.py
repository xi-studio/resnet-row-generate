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

radar_name = "../data/AZ9010_256/Z_RADR_I_Z9010_%s_P_DOR_SA_R_10_230_15.010_clean.png"

rain_list = []

img_one_hot = np.eye(256)

with open('train.csv') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        rain_list.append(row)

def default_loader(path):
    img = io.imread(path)

    img = img/255.0
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, axis=0)
    img = img.type(torch.FloatTensor)

    return img 

def predict_y(path):
    img = io.imread(path)
    y = np.round(img/16.0)
    c = np.random.randint(256) 
    y = torch.from_numpy(y[c])
    y = y.type(torch.LongTensor)
    c = img_one_hot[c]
    c = torch.from_numpy(c)
    c = c.type(torch.FloatTensor)
    
    return y, c 

class Radars(Dataset):
    def __init__(self,transform=None):
        super(Radars, self).__init__()
        self.radar = rain_list

    def __getitem__(self, index):
        img0 = default_loader(radar_name % self.radar[index][0])
        img1 = default_loader(radar_name % self.radar[index][1])
        y, c = predict_y(radar_name % self.radar[index][2])
        img  = torch.cat((img0, img1), axis=1)

        return img, y, c


    def __len__(self):
        return 10000
