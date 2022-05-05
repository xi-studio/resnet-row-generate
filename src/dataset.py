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

radar_name = "../data/AZ9010_256_h5/Z_RADR_I_Z9010_%s_P_DOR_SA_R_10_230_15.010_clean.hdf5"

rain_list = []
with open('train.csv') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        rain_list.append(row)

def default_loader(path):
    f = h5py.File(path, 'r')
    img = f['img'][:]
    f.close()
    img = torch.from_numpy(img)

    min_encodings = torch.zeros(img.shape[0], 1024).to(img)
    min_encodings.scatter_(1, img, 1)
    
    min_encodings = min_encodings.type(torch.FloatTensor)
    y = min_encodings.view(-1, 16, 16, 1024)
    y = y.permute(0, 3, 1, 2).contiguous()
    y = y.squeeze(0)
     
    return y, img 


class Radars(Dataset):
    def __init__(self,transform=None):
        super(Radars, self).__init__()
        self.radar = rain_list

    def __getitem__(self, index):
        idxa = radar_name % (self.radar[index][0])
        idxb = radar_name % (self.radar[index][1]) 
        idxc = radar_name % (self.radar[index][11]) 
        
        img_a, _ = default_loader(idxa)
        img_b, _ = default_loader(idxb)
        _, target = default_loader(idxc)

        return img_a, img_b, target


    def __len__(self):
        return 30000
