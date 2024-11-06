import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torchvision import transforms
from src.lightglue.superpoint import SuperPoint
import torch 
from src.datasets.homographies import Homography, CustomLoader

root_dir = "data/SpaceNet"
transform = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Homography(root_dir, transform=transform)
descriptor = SuperPoint(max_num_keypoints=256).eval()
dataloader =  CustomLoader(dataset, descriptor, batch_size=2)

batch = next(iter(dataloader))

print(batch['descriptors0'].shape)
# data = dataset[0]

# img1 = data['img1'].transpose(0, 2).detach().numpy()
# img2 = data['img2'].transpose(0, 2).detach().numpy()

# fig, ax = plt.subplots(1, 2, figsize=(10, 8))

# ax[0].imshow(img1)
# ax[1].imshow(img2)

# plt.show()
# plt.imshow(batch)