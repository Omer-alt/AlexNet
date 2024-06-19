import numpy as np 
import pandas as pd 
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from shutil import move
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

import os
import random

from AlexNet.model.AlexNet import AlexNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
IMAGE_PATH_VALID = os.getenv('IMAGE_PATH_VALID')
ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
SAVE_PATH = os.getenv('SAVE_PATH')


def mapping_img_cls(IMAGE_PATH_VALID):
    """To each of the validation images, associate a class with it and return the result in the form of a dataframe. """
    
    # Get the names of the validation dataset.
    image_names_valid = os.listdir(IMAGE_PATH_VALID)
    
    # An empty list for the validation dataset labels.
    image_labels_vald = []
    for i in image_names_valid:
        # Passing the path of the xml document to enable the parsing process
        tree = ET.parse(os.path.join(ANNOTATION_PATH, i[:-5] + '.xml'))
        # getting the parent tag of the xml document
        root = tree.getroot()
        image_labels_vald.append(root[5][0].text)
        
    validation_list = {"Image_Name": image_names_valid, "class": image_labels_vald}
    validation_data_frame = pd.DataFrame(validation_list)
    validation_data_frame.to_csv(os.path.join(SAVE_PATH, 'validation_list.csv'), columns=["Image_Name", "class"], index=False)
    
    return validation_data_frame

def search_cls(df, img_name):
    """Take an image name and return the corresponding classe !"""
    selected_row = df.loc[df['Image_Name'] == img_name, "class"]
    return selected_row.values[0]

def class_mapping(mapping_path):
    """Create the dictionary for the mapping in the training data"""
    
    class_mapping_dict = {}
    for idx, line in enumerate(open(mapping_path)):
        class_mapping_dict[line[:9].strip()] = (line[9:].strip(), idx)
    return class_mapping_dict


# Recover the model parameters saved in the files during training. Deuxieme
def load_model_with_weights(weight_path, num_classes=1000):
    model = AlexNet(num_classes=num_classes)
    #Load optimizer state 
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
     
    return model


# Function to visualize filters
def visualize_filters(layer_weights, num_filters=96):
    n_filters = min(layer_weights.size(0), num_filters)  # Show up to num_filters filters
    
    num_rows = 6
    num_cols = (n_filters + num_rows - 1) // num_rows  # Compute the number of rows needed
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.5, num_rows * 2.5))

    for i in range(n_filters):
        row = i // num_cols
        col = i % num_cols
        
        filter = layer_weights[i].detach().cpu().numpy()
        filter = (filter - filter.min()) / (filter.max() - filter.min())  # Normalize filter to [0, 1]

        if filter.shape[0] == 3:  # If filter has 3 channels (e.g., RGB)
            filter = np.transpose(filter, (1, 2, 0))  # Transpose to (H, W, C)
            axs[row, col].imshow(filter)
        else:  # If filter has a single channel
            axs[row, col].imshow(filter[0], cmap='gray')
        
        axs[row, col].axis('off')

    plt.show()
    
# Function to visualize filters
def visualize_filters(layer_weights,number_chanels, num_filters=96 ):
    n_filters = min(layer_weights.size(0), num_filters)  # Show up to num_filters filters
    
    num_rows = 6
    num_cols = (n_filters + num_rows - 1) // num_rows  # Compute the number of rows needed
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.5, num_rows * 2.5))

    for i in range(n_filters):
        row = i // num_cols
        col = i % num_cols
        
        filter = layer_weights[i].detach().cpu().numpy()
        filter = (filter - filter.min()) / (filter.max() - filter.min())  # Normalize filter to [0, 1]

        if number_chanels == 3:  # If filter has 3 channels (e.g., RGB)
            filter = np.transpose(filter, (1, 2, 0))  # Transpose to (H, W, C)
            axs[row, col].imshow(filter)
        else:  # If filter has a single channel
            axs[row, col].imshow(filter[0], cmap='gray')
        
        axs[row, col].axis('off')

    plt.show()
    
# Iterate through layers and visualize filters 
def filters_learned(model, number_chanels=1):
    for name, param in model.named_parameters():
        if 'weight' in name and 'layer' in name:
            print(f"Visualizing filters for layer: {name}")
            visualize_filters(param, number_chanels)
            break  # Visualize only the first convolutional layer for brevity













