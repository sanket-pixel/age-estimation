import torch
from torch import nn
from torchvision import models


class AgeEstimation(nn.Module):
    '''
    The AgeEstimation class which defines the network for learning age prediction from faces.
    Uses a pretrained VGG16 backbone and replaces final layer with fully connected layer.
    Returns log-probabilities of the given input belonging to ages 1-100
    '''
    def __init__(self,dropout=0.5):
        super(AgeEstimation, self).__init__()
        # read pretrained model VGG16 as backbone
        vgg16 = models.vgg16(pretrained=True)
        # keep all layers except last layer
        vgg16.classifier = nn.Sequential(*[vgg16.classifier[i] for i in range(4)])
        self.backbone = vgg16
        self.fc = nn.Linear(4096, 4096)
        # add fully connected layer in the end
        self.fc_last = nn.Linear(4096, 100)
        # add softmax to convert embedding to probabilities.
        # as required, the model predicts the probability of the input face
        # belonging to a particular age. ( 0,100)
        '''
        Note that logsoftmax is used instead of softmax for numerical stability.
        '''
        self.softmax = nn.LogSoftmax(dim=1)
        # applies dropout
        self.dropout = nn.Dropout(dropout)
        # non-linear activation ReLU
        self.relu = nn.ReLU()

    def forward(self, input, mode="train"):
        # extract embedding of 4096 dimension from pretrained VGG16
        embedding = self.backbone(input).squeeze(-1).squeeze(-1)
        # if mode is eval, dont apply dropout
        if mode == "eval":
            y = self.fc_last(embedding)
        else:
            y = self.dropout(self.fc_last(embedding))
        # apply log softmax to get log-probabilities
        y = self.softmax(y)
        return y


