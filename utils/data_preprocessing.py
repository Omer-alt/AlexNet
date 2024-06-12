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

import os
import random

from dotenv import load_dotenv

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













