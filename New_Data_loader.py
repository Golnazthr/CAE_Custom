#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import glob


# In[2]:


transformations = transforms.Compose([
         transforms.Resize(256, 256),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])


# In[3]:


folder_data = glob.glob ('/Users/Golnaz/Desktop/V10F24-112_A1/*.jpg')

class CustomDataset(Dataset):
    def __init__(self, image_paths, transforms=None):   # initial logic happens like transform

        self.image_paths = image_paths
        self.transforms =  transforms

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        t_image = self.transforms(image) 
        return t_image 

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)


# In[6]:


len_data = len(folder_data)
train_size = 0.8
train_image_paths = folder_data[:int(len_data*train_size)]
test_image_paths = folder_data[int(len_data*train_size):]


# In[7]:


train_dataset = CustomDataset(train_image_paths , transformations)
test_dataset = CustomDataset(test_image_paths, transformations)


# In[8]:


len_data = len(folder_data)
print("count of dataset: ", len_data)


# In[9]:


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)


# In[10]:


train_iter = iter(train_loader)
print(type(train_iter))


# In[ ]:




