from __future__ import print_function, division
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

from PIL import Image
from sklearn.model_selection import KFold,train_test_split
from scipy.stats import multivariate_normal

from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split,SubsetRandomSampler, ConcatDataset

from torch.distributions import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.model_selection import KFold

from torchmetrics.classification import AUROC
from sklearn import metrics
from itertools import cycle


def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        

seed = 0

random_state_Kfold = seed

since = time.time()
set_seed(seed)

##################### Parameter initializaiton #####################################

cudnn.benchmark = True
plt.ion()   # interactive mode

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

k=5

width=height=9
# width=height=512
#width=height=128
#batch_size = 4 # best so far
batch_size = 64
validation_split = 0.25
shuffle_dataset = True
train_ratio = 0.8
#num_epochs = 11 # best so far
num_epochs = 100


##################### Data generation #####################################
X_length = 10000
X = torch.empty(X_length,1,9,9)
Y = torch.empty(X_length,2)
for i in range(X_length):
    
    A = torch.arange(1., 82.)
    A = A.resize(9,9)
    
    A_11 = torch.normal(1, 0.5, size=(3, 3))
    A_22 = torch.normal(1, 0.5, size=(3, 3))
    A_33 = torch.normal(1, 0.5, size=(3, 3))
    
    # A11=A22=A33=A13=A31=N（1, .5）
    A_13 = torch.normal(1, 0.5, size=(3, 3))
    A_31 = torch.normal(1, 0.5, size=(3, 3))
    
    A_12 = torch.normal(0, 1, size=(3, 3))
    # A_13 = torch.normal(0, 1, size=(3, 3))
    A_21 = torch.normal(0, 1, size=(3, 3))
    A_23 = torch.normal(0, 1, size=(3, 3))
    # A_31 = torch.normal(0, 1, size=(3, 3))
    A_32 = torch.normal(0, 1, size=(3, 3))
    
    A[0:3,0:3] = A_11
    A[3:6,3:6] = A_22
    A[6:9,6:9] = A_33
    
    A[0:3,3:6] = A_12
    A[0:3,6:9] = A_13
    
    A[3:6,0:3] = A_21
    A[3:6,6:9] = A_23
    
    A[6:9,0:3] = A_31
    A[6:9,3:6] = A_32
    ##########################################################
    sigma = torch.matmul(torch.tensor([[1.,0.],[0.,2.]]), 
                         torch.matmul(torch.tensor([[1.,0.7],[0.7,1.]]), torch.tensor([[1.,0.],[0.,2.]])))
    dist = multivariate_normal.MultivariateNormal(loc=torch.zeros(2), 
                                                          covariance_matrix=sigma)
    e1, e2 = dist.sample()
    ##########################################################
    nnTanh = nn.Tanh()
    S1 = nnTanh(A_11)
    S1 = S1.sum()
    
    S2 = A_22.sum()
    
    S3 = nnTanh(A_33.sum())
    
    y1 = S1 + S2 + S3 + e1
    ##########################################################
    S1_s = nnTanh(A_31.sum())
    
    S2_s = A_22.sum()
    
    S3_s = nnTanh(A_13).sum()
    
    y2 = S1_s + S2_s + S3_s + e2
    ##########################################################
    X[i] = A
    Y[i] = torch.cat((y1.unsqueeze(0),y2.unsqueeze(0)),0)


ds_XL = TensorDataset(X.clone().detach().requires_grad_(True).float(),torch.tensor(Y).float())

PATH_ds = f'./ds_XL_seed{seed}.pt'
torch.save(ds_XL, PATH_ds)

ds_XL = torch.load(PATH_ds)



##################### useful functions #####################################

def get_test_loss(net, criterion, data_loader):
  """A simple function that iterates over `data_loader` to calculate the overall loss"""
  net.eval()
  testing_loss = []
  with torch.no_grad():
    for data in data_loader:
      inputs, labels = data
      # get the data to GPU (if available)
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = net(inputs)
      # calculate the loss for this batch
      loss = criterion(outputs, labels)
      # add the loss of this batch to the list
      testing_loss.append(loss.item())
  # calculate the average loss
  return sum(testing_loss) / len(testing_loss)

def test_stage(net, test_loader):
    net.eval()
    running_mae = [0,0]
    running_mse = [0,0]
    error = [0,0]
    squared_error = [0,0]
    mse = [0,0]
    rmse = [0,0]
    mae = [0,0]
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            
            error[0] = torch.abs(outputs[:,0] - labels[:,0]).sum().data.item()
            error[1] = torch.abs(outputs[:,1] - labels[:,1]).sum().data.item()
            squared_error[0] = ((outputs[:,0] - labels[:,0])*(outputs[:,0] - labels[:,0])).sum().data.item()
            squared_error[1] = ((outputs[:,1] - labels[:,1])*(outputs[:,1] - labels[:,1])).sum().data.item()
            running_mae[0] += error[0]
            running_mae[1] += error[1]
            running_mse[0] += squared_error[0]
            running_mse[1] += squared_error[1]
  
    mse[0] = running_mse[0]/(len(test_loader)*batch_size)
    mse[1] = running_mse[1]/(len(test_loader)*batch_size)
    rmse[0] = np.math.sqrt(running_mse[0]/(len(test_loader)*batch_size))
    rmse[1] = np.math.sqrt(running_mse[1]/(len(test_loader)*batch_size))
    mae[0] = running_mae[0]/(len(test_loader)*batch_size)
    mae[1] = running_mae[1]/(len(test_loader)*batch_size)
  
    print('test mse = ',mse)
    print('test rmse = ',rmse)
    print('test mae = ',mae)
      
    return mse,rmse,mae




def train_model(model, criterion, optimizer, scheduler,dl_train,dl_val,dl_test, num_epochs=25):
   
    best_model_wts = copy.deepcopy(model.state_dict())
    least_loss = 10000000.0
    best_epoch_num = 0
    train_loss_d = []
    validating_loss_d = []
    testing_loss_d = []
    n_train = int(len(dl_train))

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        model.train()
        
        running_loss = 0.0
        running_loss_temp = 0.0
        for i, data in enumerate(dl_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                #_, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()# * inputs.size(0)
            running_loss_temp += loss.item()
            
            if i % 20 == 19:    # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss_temp / 20:.3f}')
                running_loss_temp = 0.0
                
                avg_train_loss2 = get_test_loss(model, criterion, dl_train)
                train_loss_d.append(avg_train_loss2)
                
                avg_val_loss2 = get_test_loss(model, criterion, dl_val)
                validating_loss_d.append(avg_val_loss2)
                
                avg_test_loss2 = get_test_loss(model, criterion, dl_test)
                testing_loss_d.append(avg_test_loss2)
                
                model.train()
            
        scheduler.step()
        
        epoch_loss = running_loss / n_train

        print(f' Loss: {epoch_loss:.4f}')
        
        avg_train_loss_epoch = get_test_loss(model, criterion, dl_train)
        avg_val_loss_epoch = get_test_loss(model, criterion, dl_val)
        avg_test_loss_epoch = get_test_loss(model, criterion, dl_test)
        
        if  avg_val_loss_epoch < least_loss:
            least_loss = avg_val_loss_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch_num = epoch
            
        print("Epoch:{}/{} \
              AVG Training Loss:{:.3f} \
              AVG Validation Loss:{:.3f} \
              AVG Testing Loss:{:.3f} \
              ".format(
              epoch + 1,
              num_epochs,
              avg_train_loss_epoch,
              avg_val_loss_epoch,
              avg_test_loss_epoch,
              ))


        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best loss: {least_loss:4f}')
    print(f'Best epoch: {best_epoch_num:4f}')
    
    plt.plot(train_loss_d, label="training set")
    plt.plot(validating_loss_d, label="validation set")
    plt.plot(testing_loss_d, label="testing set")
    # make the legend on the plot
    plt.legend()
    plt.title("The MSE loss of the train/val/test data")
    plt.show()

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

##################### Neural networks for simulation #####################################
  
class Sim2Net(nn.Module):
    def __init__(self):
        super(Sim2Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 9, kernel_size=3)
        self.conv2 = nn.Conv2d(9, 18, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(450, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x), 2)
        x = F.relu(self.conv2_drop(self.conv2(x)))
        
        x = x.view(-1, 450)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #return F.log_softmax(x)
        return x


######################################################################
############  K fold training
######################################################################
# K fold
kfold =KFold(n_splits=k,shuffle=True,random_state = random_state_Kfold)

history = {'mse': [], 'rmse': [],'mae':[]}

for fold, (train_idx,test_idx) in enumerate(kfold.split(np.arange(len(ds_XL)))):
    print('=========================================')
    print('Fold {}'.format(fold + 1))
    print('=========================================')
    
    split = int(np.floor(validation_split * len(train_idx)))
    if shuffle_dataset:
        np.random.shuffle(train_idx)
    train_indices, val_indices = train_idx[split:], train_idx[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(ds_XL, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(ds_XL, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(ds_XL, batch_size=batch_size, sampler=test_sampler)
    
    PATH_dl_train = f'./dl_train_seed{seed}_fold{fold}.pt'
    torch.save(train_loader, PATH_dl_train)
    PATH_dl_val = f'./dl_val_seed{seed}_fold{fold}.pt'
    torch.save(val_loader, PATH_dl_val)
    PATH_dl_test = f'./dl_test_seed{seed}_fold{fold}.pt'
    torch.save(test_loader, PATH_dl_test)
    
    # network参数

    net = Sim2Net().to(device)

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    net = train_model(net, criterion, optimizer, exp_lr_scheduler,
                           train_loader,val_loader, test_loader, num_epochs)
    

    PATH = f'/root/Sim RR 2NN seed {seed}/resnet_simulation_regreg_seed{seed}_fold_{fold}.pth'
    # PATH = f'./resnet_simulation_regreg_seed{seed}_fold_{fold}.pth'
    
    torch.save(net.state_dict(), PATH)    
    
    
    ######################################################################
    ############  Test CNN
    ######################################################################
    net_test = Sim2Net().to(device)
    net_test.load_state_dict(torch.load(PATH))
    mse,rmse,mae = test_stage(net_test, test_loader)
    
    history['mse'].append(mse)
    history['rmse'].append(rmse)
    history['mae'].append(mae)

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

pd.DataFrame(history).to_csv(f'resnet_simulation_regreg_seed{seed}_results.csv', index=False)


















































