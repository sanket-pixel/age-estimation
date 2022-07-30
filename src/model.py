import torch
from torch import nn
from torchvision import models
import numpy as np
import os


class AgeEstimation(nn.Module):

    def __init__(self):
        super(AgeEstimation, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        vgg16.classifier =  nn.Sequential(*[vgg16.classifier[i] for i in range(4)])
        self.backbone = vgg16
        # self.backbone.requires_grad_(False)
        self.fc = nn.Linear(4096,4096)
        self.fc_last = nn.Linear(4096,100)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()


    def forward(self, input, mode = "train"):
        embedding = self.backbone(input).squeeze(-1).squeeze(-1)
        if mode =="eval":
            y = self.fc_last(embedding)
        else:
            y = self.dropout(self.fc_last(embedding))
        y = self.softmax(y)
        return y


class EstimationError(nn.Module):
    def __init__(self,sigma):
        super(EstimationError,self).__init__()
        self.sigma = sigma

    def forward(self,n,mu):
        std = torch.pow((n-mu),2)/(2*self.sigma**2)
        std_exp = torch.exp(-std)
        epsilon = 1 - std_exp
        return epsilon



# def KL_Divergence_Loss(nn.Module):
