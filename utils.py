import torch
from torchsummary import summary
from thop import profile
from thop import clever_format
import os

def GCN(x):
    """GCN"""
    mean = torch.mean(x)
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return x


def get_summary(model, input_size, device=None):
    return summary(model, input_size, device=device)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSerror:
        print('Error')

def get_model_details(model):
    input = torch.randn(32, 3, 224, 224)
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params