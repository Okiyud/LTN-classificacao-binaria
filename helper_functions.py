# -*- coding: utf-8 -*-
"""Tutorial_Helper_Functions - PyTorch Version

Converted from TensorFlow to PyTorch for LTNtorch compatibility.
"""

import numpy as np
import pandas as pd
import os
import pickle
import cv2
import random 
import seaborn as sns
import logging; logging.basicConfig(level=logging.INFO)  
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imread
from itertools import product 
from collections import defaultdict 
from tqdm import tqdm 
import ltn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.io import imshow

# dictionary for easier handling: integers to absolute attributes
dict_nb_to_colours= {0: 'dark blue', 1: 'green', 2: 'red',  3: 'baby blue',  4: 'grey', 5: 'light blue'}

# -------------------------------------------------------------------------------
# function to summarize datasets
def summarize_imported_dataset_on_object_level(d):     
  print("There are",len(d), "training examples")
  print("with the following information:",list(d[0].keys()))
  x = ['object_image'] 
  y = ["color","shape"] 
  print("For the attribute color there are",len(set([x["color"] for x in d])),"possible values")
  print("For the attribute shape there are",len(set([x["shape"] for x in d])),"possible values")

# -------------------------------------------------------------------------------
# function to visualize examples
def visualize_example(example):
  fig,axs = plt.subplots(1,2)

  axs[0].title.set_text('Original Image')
  axs[0].imshow(example["original_image"], interpolation='none')
  axs[1].title.set_text('Image of Detected Object')
  axs[1].imshow(example["object_image"], interpolation='none')
  plt.show()
  print('Position of the object center in original image:',example['object_center'])
  print("Ground truth of the object's color:",dict_nb_to_colours[example['color']])
  print("Ground truth of the object's shape:",example['shape'])

# -------------------------------------------------------------------------------
# Define the BoundingBoxDataset class first
class BoundingBoxDataset(Dataset):
    def __init__(self, dataset, resize_shape=(36, 36)):
        self.dataset = dataset
        self.resize_shape = resize_shape
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Resize image
        img_resized = cv2.resize(item['object_image'], dsize=self.resize_shape, interpolation=cv2.INTER_CUBIC)
        
        # Convert to PyTorch tensor and normalize - shape: [C, H, W]
        img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Convert shape to numeric
        shape_label = 6 if item['shape'] == "circle" else 7
        
        # Return as PyTorch tensors
        return (
            torch.tensor(idx, dtype=torch.long),  # img_index
            img_tensor,                           # img_features
            torch.tensor(shape_label, dtype=torch.long),  # labels_shape
            torch.tensor(item['color'], dtype=torch.long),  # labels_color
            torch.tensor(item['object_center'], dtype=torch.float32)  # location_feature
        )

# -------------------------------------------------------------------------------
def create_pytorch_dataloaders(dataset, batch_size=32, split_thr=0.8, resize_shape=(36, 36)):
    """Create PyTorch DataLoaders from the dataset"""
    # Create dataset
    full_dataset = BoundingBoxDataset(dataset, resize_shape=resize_shape)
    
    # Calculate split
    split_samples = int(len(full_dataset) * split_thr)
    print(f"Splitting after {split_samples} samples into train + test")
    
    # Split dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [split_samples, len(full_dataset) - split_samples]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("PyTorch DataLoaders created successfully!")
    return train_loader, test_loader

    
# -------------------------------------------------------------------------------
# creation of training and test datasets - PYTORCH VERSION
def from_bounding_boxes_list_to_ds_training_and_test(dataset, batch_size=32, split_thr=0.8, resize_shape=(36, 36)):
    # Create dataset
    full_dataset = BoundingBoxDataset(dataset, resize_shape=resize_shape)
    
    # Calculate split
    split_samples = int(len(full_dataset) * split_thr)
    print(f"we split after {split_samples} samples into train + test")
    
    # Split dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [split_samples, len(full_dataset) - split_samples]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("PyTorch DataLoaders for LTN are prepared")
    return train_loader, test_loader

# -------------------------------------------------------------------------------
class CNN_simple(nn.Module):
    def __init__(self, n_classes, img_size=[36, 36]):
        super(CNN_simple, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x, class_label):
        """
        Proper LTN predicate implementation:
        - Takes image x AND class_label as separate inputs
        - Returns a single truth value in [0,1] range for each (x, class_label) pair
        """
        # Handle LTNObject inputs
        if hasattr(x, 'value'):
            x = x.value
        if hasattr(class_label, 'value'):
            class_label = class_label.value
            
        # Process image through CNN
        features = self.conv_layers(x)
        all_logits = self.classifier(features)  # shape: [batch_size, n_classes]
        
        # class_label should be a tensor of class indices [batch_size, 1] or [batch_size]
        # Ensure it's the right shape for indexing
        if class_label.dim() > 1:
            class_label = class_label.squeeze()
            
        # Get the logits for the specified classes
        batch_indices = torch.arange(all_logits.shape[0])
        selected_logits = all_logits[batch_indices, class_label.long()]
        
        # Apply sigmoid to get probability in [0,1] range
        return torch.sigmoid(selected_logits)
        
# -------------------------------------------------------------------------------
# Simple model for left_of relation - PYTORCH VERSION
class Simple_keras_with_concatentation_left_of(nn.Module):
    def __init__(self, input_size=36*36*3*2):
        super(Simple_keras_with_concatentation_left_of, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 5),
            nn.ELU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, dummy_arg=None):
        """
        Modified to handle extra arguments from LTN
        """
        # Handle LTNObject input
        if hasattr(x, 'value'):
            x = x.value
        
        # Ignore any extra arguments, just process the main input
        x = x.view(x.size(0), -1)
        x = x / 150.0
        return self.network(x)
        
# -------------------------------------------------------------------------------
# Simple model for most_left relation - PYTORCH VERSION  
class Simple_keras_with_concatentation_most_left(nn.Module):
    def __init__(self, input_size=36*36*3*2):
        super(Simple_keras_with_concatentation_most_left, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 5),
            nn.ELU(),
            nn.Linear(5, 1),
            nn.Sigmoid()  # returns one value in [0,1]
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten all except batch dimension
        x = x / 150.0  # Normalization
        return self.network(x)

# -------------------------------------------------------------------------------
def create_absolute_attribute_predicate(cnn_model):
    """Create an LTN-compatible predicate from a CNN model for absolute attributes"""
    class AbsoluteAttributePredicate(nn.Module):
        def __init__(self, cnn_model):
            super().__init__()
            self.cnn_model = cnn_model
            
        def forward(self, x, class_label):
            # Handle LTNObject inputs
            if hasattr(x, 'value'):
                x = x.value
            if hasattr(class_label, 'value'):
                class_label = class_label.value
                
            # Get all class logits from CNN - PASS BOTH ARGUMENTS
            all_logits = self.cnn_model(x, class_label)  # Fixed: pass class_label as second argument
            
            # class_label should be tensor of shape [batch_size, 1] or [batch_size]
            # Make sure it's the right shape for indexing
            if class_label.dim() > 1:
                class_label = class_label.squeeze()
                
            # Get indices for batch and class
            batch_indices = torch.arange(all_logits.shape[0])
            class_indices = class_label.long()
            
            # Select the logits for the specified classes
            selected_logits = all_logits[batch_indices, class_indices]
            
            # Apply sigmoid to get probability in [0,1] range
            return torch.sigmoid(selected_logits)
    
    return ltn.Predicate(AbsoluteAttributePredicate(cnn_model))
    
'''
# Simple CNN model - PYTORCH VERSION
class CNN_simple(nn.Module):
    def __init__(self, n_classes, img_size=[36, 36], use_sigmoid_for_ltn=False):
        super(CNN_simple, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
        self.use_sigmoid_for_ltn = use_sigmoid_for_ltn
        
    def forward(self, x, class_label=None):
        if hasattr(x, 'value'):
            x = x.value
            
        x = self.conv_layers(x)
        x = self.classifier(x)
        
        # Add sigmoid activation for LTN predicate usage
        if self.use_sigmoid_for_ltn:
            x = torch.sigmoid(x)
            
        return x

# ----------------------------------------------------------
class CNN_simple(nn.Module):
    def __init__(self, n_classes, img_size=[36, 36]):
        super(CNN_simple, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x, class_label=None):
        # Handle LTNObject input
        if hasattr(x, 'value'):
            x = x.value
            
        # Process through CNN
        features = self.conv_layers(x)
        logits = self.classifier(features)
        
        # If class_label is provided, select the corresponding class probability
        if class_label is not None:
            # Handle LTNObject for class_label
            if hasattr(class_label, 'value'):
                class_label = class_label.value
                
            # Ensure class_label is integer indices
            if class_label.dtype == torch.float32:
                class_label = class_label.long()
                
            # Get the logits for the specified class
            # class_label should be shape [batch_size] with class indices
            selected_logits = logits[torch.arange(logits.shape[0]), class_label]
            
            # Apply sigmoid to get probability in [0,1] range
            return torch.sigmoid(selected_logits)
        
        # If no class_label, return all logits (for training)
        return logits
'''