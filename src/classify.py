import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import Radars
from model import Encoder
import torchvision.models as models

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(1, 3, 1)
        self.l2 = models.resnet152()
        self.l2.fc = nn.Linear(2048, 256 * 16)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_rec = self(x)
        loss = F.cross_entropy(y_rec.reshape(-1, 16), y.reshape(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_rec = self(x)
        loss = F.cross_entropy(y_rec.reshape(-1, 16), y.reshape(-1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=4.5e-05, betas=(0.5, 0.9))


dataset = Radars(transform=transforms.Compose([transforms.ToTensor()]))
train, val = random_split(dataset, [2000, 1000])

#trainer = pl.Trainer(gpus=-1, accelerator="dp")
trainer = pl.Trainer(gpus=[3,], max_epochs=1000)
model = LitModel()

train_loader = DataLoader(train, batch_size=32, num_workers=8)
val_loader = DataLoader(val, batch_size=32, num_workers=8)

trainer.fit(model, train_loader, val_loader)
