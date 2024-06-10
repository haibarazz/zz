import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch.nn as nn 
import requests
import zipfile
from pathlib import Path 
import random
from PIL import Image

class mydataset(Dataset):
    def __init__(self,data_dir,target_dir,transform=None):
        self.paths= list(Path(data_dir).glob("*.png"))
        labels = []
        with open(target_dir, 'r') as file:
            for line in file:
                line = line.strip()
                line_parts = int(line.split()[1])
                labels.append(line_parts)
        print(len(labels))
        self.y = torch.tensor(labels)
        self.transforms = transform

    def load_image(self, index):
        image_path = self.paths[index]
        return Image.open(image_path)     

    def __getitem__(self, index):
        img = self.load_image(index)
        label = self.y[index]
        if self.transfroms:
            return self.transfroms(img),label
        else:
            return img, label
    def __getitem__(self, index):
        img = self.load_image(index)
        label = self.y[index]
        if self.transforms is not None:  
            return self.transforms(img), label
        else:
            return img, label
    def __len__(self):
        return len(self.paths)
# train_y = "F:\python\机器学习\Data\mnist_m_train_labels.txt"
# train_x = "F:\python\机器学习\Data\mnist_m_train"
# test_y = "F:\python\机器学习\Data\mnist_m_test_labels.txt"
# test_x = "F:\python\机器学习\Data\mnist_m_test"
# train_dataset = mydataset(train_x,train_y,transform=data_transform)
# test_dataset = mydataset(test_x,test_y,transform=data_transform)




    