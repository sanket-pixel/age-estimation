import os

import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import cv2


def get_data(path):
    '''
    Reads the file containing filename and target maps.
    :param path: Path of the file
    :return: pandas dataframe
    '''
    return pd.read_csv(path)

def to_np(t):
    '''
    Converts tensor to numpy to visualize and debug
    :param t:
    :return: numpy array
    '''
    return t.permute(1, 2, 0).cpu().numpy()

def get_transform(aug):
    '''
    Returns the transform object based on the augmentation mode.
    :param aug: Boolean flag indicating if augmentation is active or not
    :return: transform object
    '''
    IMG_H = 224
    IMG_W = 224
    # if augmentation is on, then apply selected augmentations, else just apply resizing and normalization
    if aug:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(size=(IMG_H,IMG_W),
                                                          interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(p=0.3),
                                        transforms.GaussianBlur(kernel_size=3),
                                        transforms.RandomAdjustSharpness(sharpness_factor=2),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(IMG_H,IMG_W), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform


class Face(Dataset):
    '''
    Dataset class for face.
    '''
    def __init__(self, path, mode,aug=False):
        self.data = get_data(path)
        self.mode = mode
        self.aug = aug
        self.transform = get_transform(self.aug)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get filename
        file_name = self.data.iloc[idx]['file_name']
        # get target
        age = torch.Tensor([self.data.iloc[idx]['age']-1]).long()
        # read image
        img = cv2.imread(os.path.join('data/utk_sample',file_name))
        # apply transform to image
        img = self.transform(img)
        return img,age






