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
import torchvision.models as models

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.l1(x)

    def training_step(self, batch, batch_idx):
        x = batch
        print(x.shape)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, x)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


dataset = Radars(transform=transforms.Compose([transforms.ToTensor()]))
train, val = random_split(dataset, [800, 200])

#trainer = pl.Trainer(gpus=-1, accelerator="dp")
trainer = pl.Trainer(gpus=[0,], max_epochs=1000)
model = LitModel()

train_loader = DataLoader(train, batch_size=10, num_workers=8)
val_loader = DataLoader(val, batch_size=10, num_workers=8)

trainer.fit(model, train_loader, val_loader)
