import os 
import random 
import torch 
from torch.utils.data import Dataset, DataLoader

from .spacenet import SpaceNet

def knn(p1, p2, k = 2):
    distances = torch.cdist(p1, p2, p=2)  # Shape (n, m)
    knn_distances, knn_indices = torch.topk(distances, k, dim=1, largest=False)
    return knn_distances, knn_indices

class Homography(Dataset):  #Datset object to augment the raw images
    def __init__(self, root_dir, train= True, no_udm=True, transform = None):
        self.dataset  = SpaceNet(root_dir, train=train, no_udm=no_udm, transform=transform)
        
    def __len__():
        return len(SpaceNet)
    
    def __getitem__(self, idx):
        images = self.dataset[idx]
        img1, img2 = random.sample(images, 2)
        homography = []         ###

        return {'img1': img1, 'img2': img2, 'homography': homography}  #homograhy:  1 -> 2

class CustomLoader(DataLoader):
    def __init__(self, dataset, descriptor, batch_size = 16, shuffle = True) -> None:
        super.__init__(dataset, collate_fn = self.custom_collate, batch_size = batch_size, shuffle=shuffle )
        self.descriptor = descriptor           
    
    def custom_collate(self, data): 
        '''input format: List of Tuples
                ({img1, img2, homography}, .... )
            output format:
                {img1, img2, keypoints0 , keypoints1, descriptors1, descriptors2, gt_assingment}
        '''
        img1, img2, homographies = zip(*data)
        feats0 = self.descriptor({'image': img1})   #m points
        feats1 = self.descriptor({'image': img2})   #n points

        transformed_kp = torch.bmm(feats0['keypoints'], homographies)
        distances = torch.cdist(transformed_kp, feats1['keypoints'] )  # -> b x m x n 

        min_indices = torch.argmin(distances, dim=2)  #_-> b x m
    
        gt_assingment = torch.zeros_like(distances)
        
        # Set the minimum distance index in each row of each batch to 1
        batch_indices = torch.arange(gt_assingment.size(0)).unsqueeze(-1)  # Shape (batch_size, 1)
        row_indices = torch.arange(gt_assingment.size(1)).unsqueeze(0)  # Shape (1, m)
        gt_assingment[batch_indices, row_indices, min_indices] = 1

        batch = {
            "keypoints0": feats0['keypoints'],       
            "keypoints1": feats1['keypoints'], 
            "descriptors0": feats0['descriptors'], 
            "descriptors1": feats1['descriptors'],
            "view0": {"images_size" : [400, 800]},    #need to change
            "view1": {"images_size" : [400, 800]},
            "gt_assingment":  gt_assingment
        }

        return batch 
        
        

    
