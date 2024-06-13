import os
import torch

import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from AlexNet.datasets.ImageNetDataset import ImageNetDataset
from AlexNet.model.AlexNet import AlexNet
from AlexNet.utils.data_preprocessing import class_mapping, mapping_img_cls

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Retrieve environment variables
IMAGE_PATH_VALID = os.getenv('IMAGE_PATH_VALID')
ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
IMAGE_PATH_TRAIN = os.getenv('IMAGE_PATH_TRAIN')
MAPPING_PATH = os.getenv('MAPPING_PATH')
SAVE_PATH = os.getenv('SAVE_PATH')




# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    
    # Creating of mapping dataframe to get the images and classes
    data_frame = mapping_img_cls(IMAGE_PATH_VALID)
    
    # Creating of mapping dictionaries to get the image classes
    class_mapping_dict = class_mapping(MAPPING_PATH)
    
    # Create dataset instances
    train_dataset = ImageNetDataset( class_mapping_dict, IMAGE_PATH_TRAIN, transform, )
    val_dataset = ImageNetDataset( class_mapping_dict, IMAGE_PATH_VALID, transform, data_frame)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Define loss and optimizer
    model = AlexNet(num_classes=len(class_mapping_dict)).to(device)
    criterion = nn.CrossEntropyLoss()
    

if __name__ == "__main__":
    main()




