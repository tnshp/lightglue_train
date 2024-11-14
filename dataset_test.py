import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torchvision import transforms
from src.lightglue.superpoint import SuperPoint

import torch 

from src.datasets.homographies import Homography, CustomLoader
from src.viz.utils import visualize_matches
from src.lightglue import viz2d

root_dir = "data/SpaceNet"
transform = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4

dataset = Homography(root_dir, transform=transform, train=False)
descriptor = SuperPoint(max_num_keypoints=512).eval()
dataloader =  CustomLoader(dataset, descriptor, batch_size=batch_size)

batch = next(iter(dataloader))

print(batch['gt_assignment'])
# print(torch.where(batch['gt_assignment'] == 1))


for i in range(0, batch_size):
    gt_matches0 = batch['gt_matches0'][i]
    print(gt_matches0.shape)
    mask = torch.where(gt_matches0 != -1, True, False)

    kpt0 = batch['keypoints0'][i]
    kpt1 = batch['keypoints1'][i]

    m_kpts0 = kpt0[mask,:]
    m_kpts0 = m_kpts0[:, [1,0]]
    m_kpts1 = kpt1[gt_matches0[mask], :]
    m_kpts1 = m_kpts1[:, [1,0]]
    axes = viz2d.plot_images([batch['img0'][i], batch['img1'][i]])
    viz2d.plot_matches(m_kpts0, m_kpts1, lw=0.2)

    plt.show()
    # plt.clf()

    # H = batch['T'][i]
    # t = torch.tensor([H[0,2], H[1,2]])

    # print(t)

    # m_kpts0 = m_kpts0[:, [1,0]]
    # m_kpts1 = m_kpts1[:, [1,0]]
    # m_kpts0 = kpt0[mask,:] + t

    # plt.scatter(m_kpts0[:,0], m_kpts0[:,1])
    # plt.scatter(m_kpts1[:,0], m_kpts1[:,1])

    # plt.show()



# for i in range(0, batch_size):
#     gt_matches0 = batch['gt_matches0'][i]
#     mask = torch.where(gt_matches0 != -1, True, False)
    
#     H = batch['T'][i]
#     t = torch.tensor([H[0,2], H[1,2]])

#     print(t)

#     kpt0 = batch['keypoints0'][i]
#     kpt1 = batch['keypoints1'][i]

#     m_kpts0 = kpt0[mask,:] + t
#     m_kpts1 = kpt1[gt_matches0[mask], :]
#     plt.scatter(m_kpts0[:,0], m_kpts0[:,1])
#     plt.scatter(m_kpts1[:,0], m_kpts1[:,1])

#     plt.show()

# for i in range(0, batch_size):
#     visualize_matches(
#                     batch['img0'][i],       
#                     batch['img1'][i], 
#                     batch['keypoints0'][i],
#                     batch['keypoints1'][i],
#                     batch['gt_matches0'][i]
#                     )

# visualize_matches( 
#                   batch['img1'][1], 
#                   batch['img0'][1],
#                   batch['keypoints1'][1],
#                   batch['keypoints0'][1],
#                   batch['gt_matches1'][1]
#                  )    


# data = dataset[0]

# img0 = data['img0']
# img1 = data['img1']


# feats0 = descriptor({'image': img0.unsqueeze(0)})   # m points
# feats1 = descriptor({'image': img1.unsqueeze(0)}) 

# kp0 = feats0['keypoints'].squeeze(0)
# kp1 = feats1['keypoints'].squeeze(0)
# print(kp0.shape)
# kp0 = torch.cat((kp0, torch.ones(512, 1)), dim= 1)
# transformed_kp0 = torch.mm(data['homography'], kp0.T).T[:, :-1]
# # fig, ax = plt.subplots(1, 2, figsize=(10, 8))

# # ax[0].imshow(img1)
# # ax[1].imshow(img2)

# plt.scatter(transformed_kp0[:, 0], transformed_kp0[:,1], color = 'blue', s=2)
# plt.scatter(kp1[:, 0], kp1[:,1], color = 'red', s = 1)

# print(data['T'])

# plt.show()
