import os 
import random 
from omegaconf import OmegaConf
import torch 
from torch.utils.data import Dataset, DataLoader
 
from .spacenet import SpaceNet7Dataset

#to do:
#   OmegaConf

def knn(p1, p2, k = 2):  
    #p1 -> b x m x 2 , p2 -> b x n x 2 
    distances = torch.cdist(p1, p2)  # -> b x m x n 

    min_indices = torch.argmin(distances, dim=2)  #_-> b x m
    min_distances = torch.min(distances, dim=2)  #_-> b x m

    return min_indices, min_distances

class Homography(Dataset):  #Datset object to augment the raw images
    img_size = [400, 400]   
    max_descriptors = 2048
    max_overlapp = 0.5

    def __init__(self, root_dir, train= True, no_udm=True, transform = None, max_descriptors = 2048):
        self.dataset  = SpaceNet7Dataset(root_dir, train=train, no_udm=no_udm, transform=transform)
        self.max_descriptors = max_descriptors

    def __len__(self):
        return len(self.dataset)
     
    def __getitem__(self, idx):
        images = self.dataset[idx]
        img1, img2 = images[0], images[1]
        homography = []         
        ###

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
        feats0 = self.descriptor({'image': img1})   # m points
        feats1 = self.descriptor({'image': img2})   # n points

        transformed_kp = torch.bmm(feats0['keypoints'], homographies)

        min_indices, min_dist = knn(transformed_kp, feats1['keypoints']) # -> b x m 

        #add threshsolding 
        gt_assingment = torch.zeros((self.batch_size, self.dataset.max_descriptors, self.dataset.max_descriptors))
        
        # Set the minimum distance index in each row of each batch to 1
        batch_indices = torch.arange(gt_assingment.size(0)).unsqueeze(-1)  # Shape (b, 1)
        row_indices = torch.arange(gt_assingment.size(1)).unsqueeze(0)  # Shape (1, m)
        gt_assingment[batch_indices, row_indices, min_indices] = 1

        batch = {
            "keypoints0": feats0['keypoints'],       
            "keypoints1": feats1['keypoints'], 
            "descriptors0": feats0['descriptors'], 
            "descriptors1": feats1['descriptors'],
            "view0": {"images_size" : self.dataset.img_size},    
            "view1": {"images_size" : self.dataset.img_size},
            "gt_assingment":  gt_assingment
        }

        return batch
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms

    root_dir = "data/SpaceNet"
    transform = transforms.Compose([

        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = Homography(root_dir, transform=transform)
    dataloader =  CustomLoader(dataset,)