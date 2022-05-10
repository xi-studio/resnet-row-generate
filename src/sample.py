import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import Radars
import torchvision.models as models
from model import LitModel


dataset = Radars(transform=transforms.Compose([transforms.ToTensor()]))
train, val = random_split(dataset, [8000, 2000])
train_loader = DataLoader(train, batch_size=16, num_workers=8)
val_loader = DataLoader(val, batch_size=16, num_workers=8)

model = LitModel.load_from_checkpoint(checkpoint_path="lightning_logs/version_0/checkpoints/last.ckpt")
model.eval()

for batch in val_loader:
    x, y, c = batch
    y_rec = model(x, c)
    print(y_rec)
    break
