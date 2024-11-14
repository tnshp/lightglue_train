import warnings
warnings.filterwarnings("ignore")
import torch

from src.lightning.pl_lightglue import PL_lightglue
from src.lightglue.lightglue import LightGlue
from src.utils.losses import NLLLoss
from src.lightglue.superpoint import SuperPoint
from src.utils.processor import load_image, resize_image

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(config).to(device)
loss_fn = NLLLoss({})

image_paths_1 = ["data/custom/a/1.JPG", "data/custom/b/1.JPG"] #, "data/custom/c/1.JPG"]
image_paths_2 = ["data/custom/a/2.JPG", "data/custom/b/2.JPG"] #, "data/custom/c/2.JPG"]

image0 = []
image1 = []

for i in range(len(image_paths_1)):
    image0.append(load_image("data/custom/a/1.JPG", resize=[400,800]))
    image1.append(load_image("data/custom/a/2.JPG", resize=[400,800]))


image0 = torch.stack(image0)
image1 = torch.stack(image1)
 
feats0 = extractor({'image':image0.to(device)})
feats1 = extractor({'image':image1.to(device)})

feats0['image'] = image0
feats1['image'] = image1

batch = {"keypoints0": feats0['keypoints'], 
         "keypoints1": feats1['keypoints'], 
         "descriptors0": feats0['descriptors'], 
         "descriptors1": feats1['descriptors'],
         "view0": {"images_size" : [400, 800]},
         "view1": {"images_size" : [400, 800]}
         }

matches = matcher(batch)

print(matches['log_assignment'].shape)
# print(matches['matches0'])

data = {'gt_matches0':  matches['matches0'].clone() , 
        'gt_matches1':  matches['matches1'].clone(), 
        'gt_assignment': torch.ones((2048, 2048))  # m x n matrix of assignment
        }

loss, weights , _= loss_fn(matches, data)
# loss = torch.mean(losses["total"])
print(loss)
    
# print(batch['image1'].keys())
# print(feats0['keypoints'].shape, feats0['keypoint_scores'].shape, feats0['descriptors'].shape)
# print(matches['matches'])
# print(matches['matches'][0].shape, matches['scores'][0].shape,  matches['scores'][1].shape)
