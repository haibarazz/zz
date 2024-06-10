import torch
from torch import nn
import requests
import zipfile
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import datasets
from Config import *


data_transform = transforms.Compose([
    transforms.Resize(size=(28, 28)),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor() 
])

train_data = datasets.ImageFolder(root=Config.train_dir, 
                                  transform=data_transform, 
                                  target_transform=None) 
test_data = datasets.ImageFolder(root=Config.test_dir, 
                                 transform=data_transform)