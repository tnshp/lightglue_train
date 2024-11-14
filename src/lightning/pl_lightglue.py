import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

from src.utils.losses import NLLLoss
from src.lightglue.lightglue import LightGlue


class PL_lightglue(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.matcher = LightGlue(config)
        self.loss_fn = NLLLoss({})
        # self.save_hyperparameters(    )

    def training_step(self, batch, batch_idx):
      
        matches = self.matcher(batch)
        loss, weights , _= self.loss_fn(matches, batch)
        loss = torch.mean(loss)
        self.log("train_loss", loss, prog_bar=True) 
        return loss
    
    def test_step(self, batch, batch_idx):
        matches = self.matcher(batch)
        loss, weights , _= self.loss_fn(matches, batch)
        loss = torch.mean(loss)
        self.log("test_loss", loss, prog_bar=True) 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

    
    


