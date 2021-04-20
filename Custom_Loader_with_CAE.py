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
            transforms.Resize ((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
                              ]) 


# In[3]:


folder_data = glob.glob ('/Users/Golnaz/Desktop/V10F24-112_A1/*.jpg')

class CustomDataset(Dataset):
    def __init__(self, image_paths, transforms=None):   # initial logic happens like transform

        self.image_paths = folder_data
        self.transforms =  transforms

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        t_image = self.transforms(image) 
        return t_image 

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)


# In[4]:


len_data = len(folder_data)
train_size = 0.8
train_image_paths = folder_data[:int(len_data*train_size)]
test_image_paths = folder_data[int(len_data*train_size):]


# In[5]:


train_dataset = CustomDataset(train_image_paths , transformations)
test_dataset = CustomDataset(test_image_paths, transformations)


# In[6]:


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)


# In[7]:


class Autoencoder(nn.Module):    
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))        
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True))    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# In[8]:


model = Autoencoder()
print(model)


# In[9]:


# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for e in range(1, epochs+1):
    train_loss = 0.0  # monitor training loss
    
    ###################
    # train the model #
    ###################
    for data in train_loader:
        images = data                        # we are just intrested in images
        # no need to flatten images
        optimizer.zero_grad()                   # clear the gradients
        outputs = model(images)                 # forward pass: compute predicted outputs 
        loss = criterion(outputs, images)       # calculate the loss
        loss.backward()                         # backward pass
        optimizer.step()                        # perform optimization step
        train_loss += loss.item()*images.size(0)# update running training loss
            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {}'.format(e),
          '\tTraining Loss: {:.4f}'.format(train_loss))


# In[44]:


# Lets get batch of test images
dataiter = iter(test_loader)
images = dataiter.next()

output = model(images)                     # get sample outputs
images = images.numpy()                     # prep images for display
output = output.detach().numpy()           # use detach when it's an output that requires_grad
images = images[0,:,:]
output = output [0,:,:]



# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
    


# In[ ]:





# In[ ]:




