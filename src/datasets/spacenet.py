import os
import random 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SpaceNet7Dataset(Dataset):
    def __init__(self, root_dir,train= True, sample_size = 2,transform=None):
        self.root_dir = root_dir
        if train:
            self.image_dir = os.path.join(root_dir, 'SN7_buildings_train', 'train')
        else:
            
            self.image_dir = os.path.join(root_dir, 'SN7_buildings_test_public', 'test_public')
        self.sample_size = sample_size
        self.transform = transform
        self.image_paths = []
        
        # Go through each sub-folder in the root directory
        for folder in os.listdir(self.image_dir):
            folder_path = os.path.join(self.image_dir, folder, "images" if train else "images_masked")
            if os.path.isdir(folder_path):
                # Collect image paths in the "images" folder of the current location
                image_files = sorted(os.listdir(folder_path))
                image_paths = [os.path.join(folder_path, img) for img in image_files]
                
                # Only add to dataset if there are exactly 24 images or more
                if len(image_paths) >= 24:
                    self.image_paths.append(image_paths[:24])  # Take the first 24 images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        # Load 24 images for the specific location
        images = []
        img_paths = random.sample(self.image_paths[idx], self.sample_size) #sample 2 images
        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        images_tensor = torch.stack(images)  # Shape: (S, C, H, W)
        
        return images_tensor

# Example usage
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    root_dir = "data/SpaceNet"
    transform = transforms.Compose([

        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = SpaceNet7Dataset(root_dir=root_dir, train=False,transform=transform)
    print(len(dataset))
    images = dataset[0]

    print(images.shape)
    image = images[0].transpose(0,2)
    image = image.detach().cpu().numpy()
    plt.imshow(image)
    plt.show()
    # data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # # Example loop over the data loader
    # for batch in data_loader:
    #     print(batch.shape)  # Expected shape: (batch_size, 24, C, H, W)
    #     break
