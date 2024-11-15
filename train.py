import lightning as L
import torch
import argparse

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torchvision import transforms

from src.lightglue.superpoint import SuperPoint
from src.lightning.pl_lightglue import PL_lightglue
from src.datasets.homographies import Homography, CustomLoader

parser = argparse.ArgumentParser(description="lightglue training script")

# Add arguments
parser.add_argument('--max_epochs', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=16)

# Parse the arguments
args = parser.parse_args()


root_dir = "data/SpaceNet"
transform = transforms.Compose([
    transforms.ToTensor()
])

config = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    } 


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_set = Homography(root_dir, transform=transform)
test_set = Homography(root_dir, transform=transform, train=False)

descriptor = SuperPoint(max_num_keypoints=512).eval()

train_loader =  CustomLoader(train_set, descriptor, batch_size=args.batch_size)
test_loader =  CustomLoader(test_set, descriptor, batch_size=args.batch_size)

matcher = PL_lightglue(config=config)
trainer = L.Trainer(limit_train_batches=100, max_epochs=args.max_epochs, accelerator=device)
trainer.fit(model=matcher, train_dataloaders=train_loader)

# trainer.test(model=matcher, dataloaders=test_loader) 
#RuntimeError: Sizes of tensors must match except in dimension 2. Expected size 1 but got size 4 for tensor number 1 in the list.
# Testing DataLoader 0:  75%|███████▌  | 3/4 [00:03<00:01,  0.76it/s]

#tensorboard not logging