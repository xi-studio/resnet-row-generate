import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from dataset_v0 import Radars
from model_resnet import LitModel
import torchvision.models as models


dataset = Radars(transform=transforms.Compose([transforms.ToTensor()]))
train, val = random_split(dataset, [8000, 2000])

#trainer = pl.Trainer(gpus=-1, accelerator="dp")
trainer = pl.Trainer(gpus=[3,], max_epochs=1000)
model = LitModel()

train_loader = DataLoader(train, batch_size=16, num_workers=8)
val_loader = DataLoader(val, batch_size=16, num_workers=8)

trainer.fit(model, train_loader, val_loader)
