#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


# convert data to transformed Version

# load the training and test datasets


# In[4]:


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.relu = nn.ReLU()
        
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool1(self.relu(self.conv1(x)))
        out = self.pool2(self.relu(self.conv2(out)))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.pool5(self.relu(self.conv5(out)))
        out = out.view(out.size(0), -1)
        
        #out = self.drop_out(out)
        out = self.drop_out(out)
        out = self.relu(self.fc1(out))
        out = self.drop_out(out)
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out
    


# In[5]:


model = ConvNet()
print  (model)


# In[ ]:


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
        images, _ = data                        # we are just intrested in images
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

