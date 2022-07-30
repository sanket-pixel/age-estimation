import os

import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import cv2


def get_data(path):
    return pd.read_csv(path)

def to_np(t):
    return t.permute(1, 2, 0).cpu().numpy()

def get_transform( mode="train"):
    IMG_H = 224
    IMG_W = 224
    if mode == "train":
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(size=(IMG_H,IMG_W),
                                                          interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    elif mode == "validation":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(IMG_H,IMG_W), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform


class Face(Dataset):
    def __init__(self, path, mode):
        self.data = get_data(path)
        self.mode = mode
        self.transform = get_transform(self.mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx]['file_name']
        age = torch.Tensor([self.data.iloc[idx]['age']-1]).long()
        img = cv2.imread(os.path.join('data/utk_sample',file_name))
        img = self.transform(img)
        return img,age






