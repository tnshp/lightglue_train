import os 
import numpy as np
import random 
from omegaconf import OmegaConf
import torch 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from src.datasets.spacenet import SpaceNet7Dataset


def knn(p1, p2, k = 2):  
    #p1 -> b x m x 2 , p2 -> b x n x 2 
    distances = torch.cdist(p1, p2)  # -> b x m x n 

    min_indices = torch.argmin(distances, dim=2)  #_-> b x m
    min_distances = torch.min(distances, dim=2)  #_-> b x m

    return min_indices, min_distances

def crop_patch(image, center, size):
    
    C, H, W = image.shape
    patch_height, patch_width = size[0], size[1]
    center_y, center_x = center[0], center[1]

    # Calculate the starting and ending indices for the crop, ensuring they are within bounds
    start_y = max(center_y - patch_height // 2, 0)
    start_x = max(center_x - patch_width // 2, 0)
    end_y = min(start_y + patch_height, H)
    end_x = min(start_x + patch_width, W)
    
    # Adjust start indices if end indices go out of bounds
    start_y = max(end_y - patch_height, 0)
    start_x = max(end_x - patch_width, 0)

    # Crop the patch
    patch = image[:, start_y:end_y, start_x:end_x]
    return patch

class Homography(Dataset):  #Datset object to augment the raw images
    default_conf = {
        'name': 'spacenet',
        'map_size': [1024, 1023],
        'img_sizes': [[128, 128], [256, 256], [512, 512]],
        'img_resize': [256, 256],
        'translate_range': [0.1, 0.4],    # max in between 0.25 to 0.5, defines how to transalte the centre of second imgae w.r.t. to first        
        'max_descriptors': 256,
        'min_overlap': 0.5,
    }

    def __init__(self, root_dir, conf = {},  train= True, transform = None):
        self.dataset  = SpaceNet7Dataset(root_dir, train=train, transform=transform)
        self.conf = OmegaConf.merge(self.default_conf, conf)

    def __len__(self):
        return len(self.dataset)
     
    def __getitem__(self, idx):
        images = self.dataset[idx]
        img1, img2 = images[0], images[1]

        # Homography 
        #choose image sizes
        idx1 = random.randint(0, len(self.conf.img_sizes) - 1)
        size1 = np.array(self.conf.img_sizes[idx1])
        idx2 = max(0, idx1 if np.random.rand(1) > 0.5 else idx1 - 1)
        print('idx2: ', idx2)
        size2 = np.array(self.conf.img_sizes[idx2] ) # sample same scale or 1 lvl lower

        #choose 1st image centre
        c1 = [random.randint(
            int(size1[0] /  2),
            int(self.conf.map_size[0] - size1[0] /  2)
        ),
        random.randint(
            int(size1[1] /  2),
            int(self.conf.map_size[1] - size1[1] /  2)
        )]
        c1 = np.array(c1)

        in_range = False
        while(not in_range):
            t = self.conf.translate_range[0] + (self.conf.translate_range[1] - self.conf.translate_range[0]) * np.random.rand(2)
            t = t * np.where(np.random.rand(2) > 0.5, 1, -1)
            t = t * size1
            c2 = c1 + t  
            c2 = c2.astype(int)
            
            #check if in range of map
            high = np.array(self.conf.map_size)  
            low = np.array([0,0])
            in_range = (c2 >= low) & (c2 < high)  
            in_range = in_range[0] and in_range[1]
            print('in_range: ' , in_range)

        T = torch.tensor([[1,0 , -t[0]], 
                          [0,1, -t[1]],
                          [0, 0 ,1]]
                        ).float()
        S = torch.tensor([[2,0,0],[0,2,0], [0,0,1]]).float() if size1[0] >  size2[0] else torch.eye(3) 

        #add rotation
        H = torch.mm(T, S)      

        #crop and resize
        resize = transforms.Resize(self.conf.img_resize)
        print('size1: ', size1)
        img1 = crop_patch(img1, c1, size1)
        img1 = resize(img1.unsqueeze(0)).squeeze(0)
        
        img2 = crop_patch(img2, c2, size2)
        img2 = resize(img2.unsqueeze(0)).squeeze(0)

        return {'img1': img1, 'img2': img2, 'homography': H}  #homograhy:  1 -> 2

class CustomLoader(DataLoader):
    def __init__(self, dataset, descriptor, batch_size = 16, shuffle = True) -> None:
        self.descriptor = descriptor           
        super(CustomLoader, self).__init__(dataset, collate_fn = self.custom_collate, batch_size = batch_size, shuffle=shuffle )
    
    def custom_collate(self, batch): 
        '''input format: List of Tuples
                ({img1, img2, homography, img_size}, .... )
            output format:
                {img1, img2, keypoints0 , keypoints1, descriptors1, descriptors2, gt_assingment}
        '''
        data = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
        img1 = data['img1']
        img2 = data['img2']
        homographies = data['homography']

        feats0 = self.descriptor({'image': img1})   # m points
        feats1 = self.descriptor({'image': img2})   # n points

        kp1 = feats0['keypoints']
        kp1 = torch.cat((kp1, torch.ones(self.batch_size, self.dataset.conf.max_descriptors, 1)), dim= 2)
        transformed_kp1 = torch.bmm(kp1, homographies)[:,:, :-1]
    
        min_indices, min_dist = knn(transformed_kp1, feats1['keypoints']) # -> b x m 

        #remove out of bounds points
        
        gt_assingment = torch.zeros((self.batch_size, self.dataset.conf.max_descriptors, self.dataset.conf.max_descriptors))
        
        # Set the minimum distance index in each row of each batch to 1
        batch_indices = torch.arange(gt_assingment.size(0)).unsqueeze(-1)  # Shape (b, 1)
        row_indices = torch.arange(gt_assingment.size(1)).unsqueeze(0)  # Shape (1, m)
        gt_assingment[batch_indices, row_indices, min_indices] = 1

        batch = {
            "keypoints0": feats0['keypoints'],       
            "keypoints1": feats1['keypoints'], 
            "descriptors0": feats0['descriptors'], 
            "descriptors1": feats1['descriptors'],
            "view0": {"images_size" : self.dataset.conf.img_resize},    
            "view1": {"images_size" : self.dataset.conf.img_resize},
            "gt_assingment":  gt_assingment
        }

        return batch
        
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from torchvision import transforms
#     from src.lightglue.superpoint import SuperPoint
#     root_dir = "data/SpaceNet"
#     transform = transforms.Compose([

#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     dataset = Homography(root_dir, transform=transform)
#     descriptor = SuperPoint(max_num_keypoints=2048).eval().to(device)
#     dataloader =  CustomLoader(dataset, descriptor, batch_size=2)

#     batch = next(iter(dataloader))