import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(1, 3, 1)
        self.l2 = models.resnet152()
        self.l3 = nn.Sequential(
                    nn.Linear(1000 + 256 + 20, 256 * 16),
                    nn.ReLU()
                  )

    def forward(self, x, c, t):
        x = self.l1(x)
        x = self.l2(x)
        x = torch.cat((x, c, t), axis=1)
        x = self.l3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, c, t = batch
        y_rec = self(x, c, t)
        loss  = F.cross_entropy(y_rec.reshape(-1, 16), y.reshape(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, c, t = batch
        y_rec = self(x, c, t)
        loss  = F.cross_entropy(y_rec.reshape(-1, 16), y.reshape(-1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=4.5e-05, betas=(0.5, 0.9))
