import os

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from AlexNet.utils.data_preprocessing import search_cls

class ImageNetDataset(Dataset):
    def __init__(self, class_mapping_dict, root_dir, transform=None, df=None, limit=1000):
        self.root_dir = root_dir
        self.transform = transform
        self.class_mapping_dict = class_mapping_dict
        self.df = df
        
        #  To run just part of dataset
        self.limit = limit
        
        self.images = []
        self.labels = []
        
        self.images_val = []
        self.labels_val = []
        
        if self.df is not None: 
            for img_name in tqdm(os.listdir(root_dir)):
                self.images_val.append(os.path.join(root_dir, img_name))
                #chercher le nom de dossier
                label_name = search_cls(df, img_name)
                #chercher la classe correspondante
                mapping_class_to_number = class_mapping_dict[label_name][1]
                self.labels_val.append(mapping_class_to_number)
        else:
#             print("The directory: ", os.listdir(root_dir))
            for train_class in tqdm(os.listdir(root_dir)[:self.limit]):
                class_path = os.path.join(root_dir, train_class)
                for img_name in os.listdir(class_path):
                    self.images.append(os.path.join(class_path, img_name))
#                     self.labels.append(mapping_class_to_number[train_class])
                    mapping_class_to_number = class_mapping_dict[train_class][1]
                    self.labels.append(mapping_class_to_number)
            
    
    def __len__(self):
        if self.df is not None:
            return len(self.images_val)
        return len(self.images)
    
    def __getitem__(self, idx):
        
        if self.df is None: 
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
        else:
            img_path = self.images_val[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels_val[idx]
                        
        
        if self.transform:
            image = self.transform(image)
        
        return image, label