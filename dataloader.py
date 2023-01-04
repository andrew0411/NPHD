from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
from torch.utils.data import DataLoader, Subset
import torch

def get_dataset(folder_path=None, transform=None):
    dataset = datasets.ImageFolder(root=folder_path, transform=transform)
    print(f'Dataset size : {len(dataset)}')
    return dataset

def GCN(x):
    """GCN"""
    mean = torch.mean(x)
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return x

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def get_dataloader(folder_path=None, transform=None, val_split=0.2, train_batch=64):
    dataset = get_dataset(folder_path=folder_path, transform=transform)
    _dataset = train_val_dataset(dataset, val_split=val_split)


    print(len(_dataset['train']))
    train_loader = DataLoader(dataset=_dataset['train'], batch_size=train_batch, shuffle=True, num_workers=8)
    print(len(_dataset['val']))
    val_loader = DataLoader(dataset=_dataset['val'], batch_size=len(_dataset['val']))

    x, y = next(iter(train_loader))
    print(f'X : {x.shape}, y : {y.shape}')

    return train_loader, val_loader