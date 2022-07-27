import torch
from torch import nn
from torchvision import models
import numpy as np
import os


class AgeEstimation(nn.Module):

    def __init__(self):
        super(AgeEstimation, self).__init__()
        self.model = models.resnet18(pretrained=True)
        fc_inputs = self.model.fc.in_features
        self.model.fc(fc_inputs, 1)

    def forward(self, input):
        y = self.model(input)
        y = torch.round(y)
        y = torch.clip(y, 0, 100)
        return y
