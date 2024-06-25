from __future__ import print_function, division
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

import torchvision.models.segmentation
import torchvision.transforms as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

from PIL import Image
from sklearn.model_selection import KFold,train_test_split
from scipy.stats import multivariate_normal

from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split,SubsetRandomSampler, ConcatDataset

from torch.distributions import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import normal

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math

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
num_epochs = 30

##################### Data loading #####################################
PATH_ds = f'/root/Sim class_reg seed {seed}/ds_XL_seed{seed}.pt'

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
      
      classifications_labels = torch.transpose(labels[:,0].unsqueeze(0), 0, 1).to(device)
      regression_labels = torch.transpose(labels[:,1].unsqueeze(0), 0, 1).to(device)
      
      # print("classifications_labels:", classifications_labels)
      # print("regression_labels:", regression_labels)
      
      outputs_classification, outputs_regression = net(inputs)
      
      # calculate the loss for this batch
      loss = criterion(outputs_classification, outputs_regression, classifications_labels,
                             regression_labels, gamma, std_ei)
      
      # add the loss of this batch to the list
      testing_loss.append(loss.item())
      # print(loss)
  # calculate the average loss
  return sum(testing_loss) #/ len(testing_loss)

def test_stage(net, test_loader):
    net.eval()
    y_test = []
    y_score = []
    
    
    correct_classification = 0
    total_classification = 0
    total_regression_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            classifications_labels = torch.transpose(labels[:,0].unsqueeze(0), 0, 1).to(device)
            regression_labels = torch.transpose(labels[:,1].unsqueeze(0), 0, 1).to(device)
            # calculate outputs by running images through the network
            outputs_classification, outputs_regression = net(images)
            
            y_score.append(outputs_classification.cpu().detach().numpy())
            y_test.append(classifications_labels.cpu().detach().numpy())
            
            loss_classification = criterion_classification(outputs_classification, classifications_labels)
            loss_regression = criterion_regression(outputs_regression, regression_labels)
            loss = loss_classification + loss_regression
            
            total_regression_loss += loss_regression.item()
            
            predicted_classification = outputs_classification > 0.5
            total_classification += classifications_labels.size(0)
            correct_classification += (predicted_classification == classifications_labels).sum().item()
            
    y_test = np.array([t.ravel() for t in y_test])
    y_score = np.array([t.ravel() for t in y_score])
    
    """
    compute ROC curve and ROC area for each class in each fold
    """
    AUC_results = AUC_plot(y_test, y_score)
    
    ############################### 
    y_test2 = torch.flatten(torch.tensor(y_test))
    y_score2 = torch.flatten(torch.tensor(y_score))
    
    auroc = AUROC(task="binary")
    AUC_doubleCheck = auroc(y_score2, y_test2)
    print("AUC doubleCheck: ", AUC_doubleCheck.detach().cpu().numpy())
    ###############################
    
    
    Classification_accuracy = correct_classification / total_classification
    Regression_loss = total_regression_loss / len(test_loader)
    print('Testing stage: Classification accuracy: %.2f%%, Regression loss: %.3f' %
          (100 * correct_classification / total_classification, total_regression_loss / len(test_loader)))
      
    return Classification_accuracy,Regression_loss, AUC_results
      
def AUC_plot(y_test, y_score):
    """
    compute ROC curve and ROC area for each class in each fold
    """
    N_classes = 2
    fpr = dict()
    tpr = dict()
    local_roc_auc = dict()
    for i in range(N_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(np.array(y_test[:, i]),np.array(y_score[:, i]))
        local_roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    local_roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= N_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    local_roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(local_roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(local_roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(N_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(i, local_roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-1e-2, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc="lower right")
    plt.savefig(f'resnet_COPULA_ClassRegre_fold{fold}.png', dpi = 600)
    plt.show()
    
    
    #print(local_roc_auc)
    return local_roc_auc

class classReg_Loss(nn.Module):
    def __init__(self):
        super(classReg_Loss, self).__init__()
 
    def forward(self, outputs_classification, outputs_regression, classifications_labels,
                             regression_labels, gamma, std_ei):
        
        #计算 (y1 - y1_hat)^2
        yi_minus_y_ihat = (regression_labels - outputs_regression).to(device)
        yi_minus_y_ihat_sqr = torch.square(yi_minus_y_ihat)
        
        
        #计算 log内部的Phi(。。。)
        rho = gamma[1][0]
        dist_phi = normal.Normal(torch.zeros(1).cuda(), 1)
        
        log_xxx_vec =torch.empty(0).to(device)
        
        # print(outputs_classification)
        for i, p2_i in enumerate(outputs_classification,0):
            # print(p2_i, dist_phi.icdf(p2_i))
            p2_i = torch.where(p2_i < 0.0000001, 0.0000001, p2_i)
            p2_i = torch.where(p2_i > 0.9999999, 0.9999999, p2_i)
            
            
            phi_inv_p2_minus_rho_dvd_sqr = (dist_phi.icdf(p2_i) - rho * yi_minus_y_ihat[i] / std_ei) / torch.sqrt(1-torch.square(rho))

            PHI_phi_inv_p2_minus_rho_dvd_sqr = dist_phi.cdf(phi_inv_p2_minus_rho_dvd_sqr)
            # PHI_phi_inv_p2_minus_rho_dvd_sqr = torch.where(torch.logical_and(classifications_labels[i] == 1, PHI_phi_inv_p2_minus_rho_dvd_sqr < 0.0000001), 0.0000001, PHI_phi_inv_p2_minus_rho_dvd_sqr)
            PHI_phi_inv_p2_minus_rho_dvd_sqr = torch.where(PHI_phi_inv_p2_minus_rho_dvd_sqr < 0.0000001, 0.0000001, PHI_phi_inv_p2_minus_rho_dvd_sqr)
            PHI_phi_inv_p2_minus_rho_dvd_sqr = torch.where(PHI_phi_inv_p2_minus_rho_dvd_sqr > 0.9999999, 0.9999999, PHI_phi_inv_p2_minus_rho_dvd_sqr)
            log_xxx_i =  torch.where(classifications_labels[i] == 0, torch.log(1-PHI_phi_inv_p2_minus_rho_dvd_sqr), torch.log(PHI_phi_inv_p2_minus_rho_dvd_sqr))
            # print("PHI_phi_inv_p2_minus_rho_dvd_sqr:", PHI_phi_inv_p2_minus_rho_dvd_sqr)
            # print("log_xxx_i:", log_xxx_i)
            log_xxx_vec = torch.cat((log_xxx_vec,log_xxx_i),dim=0)

        loss = (torch.sum(yi_minus_y_ihat_sqr)/std_ei - torch.sum(log_xxx_vec))
        # print("phi_inv_p2_minus_rho_dvd_sqr:", phi_inv_p2_minus_rho_dvd_sqr)
        # print("PHI_phi_inv_p2_minus_rho_dvd_sqr:", PHI_phi_inv_p2_minus_rho_dvd_sqr)
        # print("log_xxx_vec:", log_xxx_vec)
        # print("loss:", loss)
        return loss  

def train_model(model, criterion, criterion_classification, criterion_regression, optimizer, scheduler,dl_train,dl_val,dl_test, num_epochs):
   
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
            
            classifications_labels = torch.transpose(labels[:,0].unsqueeze(0), 0, 1).to(device)
            regression_labels = torch.transpose(labels[:,1].unsqueeze(0), 0, 1).to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Forward pass
                outputs_classification, outputs_regression = model(inputs)
                
                loss = criterion(outputs_classification, outputs_regression, classifications_labels,
                             regression_labels, gamma ,std_ei)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
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
    plt.title("The COPULA loss of the train/val/test data")
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
        self.sigmoid = nn.Sigmoid()

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
        #print(x)
        y_classification = self.sigmoid(x[:,0])
        y_regression = x[:,1]
        #return F.log_softmax(x)
        
        y_classification = torch.transpose(y_classification.unsqueeze(0), 1, 0)
        y_regression = torch.transpose(y_regression.unsqueeze(0), 1, 0)
        
        return y_classification, y_regression




######################################################################
############  K fold training
######################################################################
# K fold
kfold =KFold(n_splits=k,shuffle=True,random_state = random_state_Kfold)

history = {'Classification_accuracy': [], 'Regression_loss': [], 'AUC_results': []}

for fold, (train_idx,test_idx) in enumerate(kfold.split(np.arange(len(ds_XL)))):
    print('=========================================')
    print('Fold {}'.format(fold + 1))
    print('=========================================')
    
    # PATH_dl_train = f'./dl_train_seed{seed}_fold{fold}.pt'
    PATH_dl_train = f'/root/Sim class_reg seed {seed}/dl_train_seed{seed}_fold{fold}.pt'
    train_loader = torch.load(PATH_dl_train)
    
    # PATH_dl_val = f'./dl_val_seed{seed}_fold{fold}.pt'
    PATH_dl_val = f'/root/Sim class_reg seed {seed}/dl_val_seed{seed}_fold{fold}.pt'
    val_loader = torch.load(PATH_dl_val)
    
    # PATH_dl_test = f'./dl_test_seed{seed}_fold{fold}.pt'
    PATH_dl_test = f'/root/Sim class_reg seed {seed}/dl_test_seed{seed}_fold{fold}.pt'
    test_loader = torch.load(PATH_dl_test)
    
    
    
    # network参数
    # PATH = f'./resnet_ClassRegre_crossVal_fold_{fold}.pth'
    # PATH = f'./resnet_ClassReg_batch32_epo30_seed123_step7_0p5_fold{fold}.pth'
    PATH = f'/root/Sim class_reg seed {seed}/resnet_simulation_classReg_seed{seed}_fold_{fold}.pth'
    net = Sim2Net().to(device)
    net.load_state_dict(torch.load(PATH))

    criterion_classification = nn.BCELoss()
    criterion_regression = nn.MSELoss()
    
    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    criterion = classReg_Loss()
    
    ######################################################################
    ############ 计算gamma
    ######################################################################
       
    ei = torch.empty(0).to(device)
    pj = torch.empty(0).to(device)
    g1 = torch.empty(0).to(device)
    
    # calculate residual for every training sample
    with torch.no_grad():
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            net.eval()
            outputs = net(inputs)
            
            pj_i = outputs[0].squeeze()
            ei_i = labels[:,1] - outputs[1].squeeze()
            g1_x = labels[:,1]
            
            ei = torch.cat((ei,ei_i),dim=0)
            pj = torch.cat((pj,pj_i),dim=0)
            g1 = torch.cat((g1,g1_x),dim=0)
    
    Phi_inverse_pj = torch.empty(0).to(device)
    dist_phi = normal.Normal(torch.zeros(1).cuda(), 1)
    for i, pj_ii in enumerate(pj, 0):
        pj_ii = torch.where(pj_ii < 0.0000001, 0.0000001, pj_ii)
        pj_ii = torch.where(pj_ii > 0.9999999, 0.9999999, pj_ii)
        Phi_inverse_pj = torch.cat((Phi_inverse_pj,dist_phi.icdf(pj_ii)),dim=0)
    
    # ei_phi_inv_pj = torch.stack((ei, Phi_inverse_pj),0)
    g1_phi_inv_pj = torch.stack((g1, Phi_inverse_pj),0)
     
    std_ei = torch.std(ei)
    # gamma = torch.corrcoef(ei_phi_inv_pj)
    gamma = torch.corrcoef(g1_phi_inv_pj)
    print('Gamma: ', gamma)
        
    ######################################################################
    ############ train
    ######################################################################  
    
    net = train_model(net, criterion, criterion_classification, criterion_regression, optimizer, exp_lr_scheduler,
                           train_loader,val_loader, test_loader,num_epochs)
    

    PATH = f'/root/Sim class_reg seed {seed}/resnet_COPULA_simulation_classReg_seed{seed}_fold_{fold}.pth'
    torch.save(net.state_dict(), PATH)    
    
    
    ######################################################################
    ############  Test CNN
    ######################################################################
    net_test = Sim2Net().to(device)

    net_test.load_state_dict(torch.load(PATH))
    Classification_accuracy,Regression_loss, AUC_results = test_stage(net_test, test_loader)
    
    history['Classification_accuracy'].append(Classification_accuracy)
    history['Regression_loss'].append(Regression_loss)
    history['AUC_results'].append(AUC_results)

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

pd.DataFrame(history).to_csv(f'resnet_COPULA_simulation_classReg_seed{seed}_results new data.csv', index=False)

























































