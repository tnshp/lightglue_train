import os
from torch.utils.data import Dataset
from PIL import Image

class SpaceNet(Dataset):
    def __init__(self, root_dir, train= True, no_udm=True, transform = None): 
        self.root_dir = root_dir
        # self.annotations = pd.read_csv(os.path.join(root_dir, 'SN7_buildings_train_csvs', 'sn7_train_ground_truth_pix.csv'))
        self.no_udm = no_udm
        self.transform = transform
        self.train =train
        self.image_dir_path = root_dir
        if train:
            self.image_dir_path = os.path.join(root_dir, 'SN7_buildings_train', 'train')
        else:
            self.image_dir_path = os.path.join(root_dir, 'SN7_buildings_test_public', 'test_public')

        self.image_folders = [name for name in os.listdir(self.image_dir_path) if os.path.isdir(os.path.join(self.image_dir_path, name))]
            
    def __len__(self):
        return len(self.image_folders)
    
    def __getitem__(self, idx):
        
        if self.train:
            image_folder = os.path.join(self.image_dir_path, self.image_folders[idx], 'images')
        else:
            image_folder = os.path.join(self.image_dir_path, self.image_folders[idx], 'images_masked')

        img_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.tif')]
       
        images = [Image.open(img_path) for img_path in img_paths]

        if self.transform:
            images = [self.transform(image) for image in images]

        return images

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dir  = './data/SpaceNet'
    dataset = SpaceNet(dir, train=False)  
    
    print(len(dataset))

    img_list = dataset[10]
    plt.imshow(img_list[0])
    plt.show()



