import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

from src.utils.losses import NLLLoss
from lightglue.lightglue import LightGlue


class PL_lightglue(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.matcher = LightGlue(config)
        self.loss_fn = NLLLoss({})

    def training_step(self, batch, data, batch_idx):
        
        #batch in format {"image0": feats0, "image1": feats1}
        matches = self.matcher(batch)
        loss, weights , _= self.loss_fn(matches, data)
        loss = torch.mean(loss)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

    
    


