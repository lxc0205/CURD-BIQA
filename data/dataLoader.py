import os
import torch
import data.folders as folders
import torchvision
import numpy as np

folder_path ={
    'csiq': '/mnt/g/datasets/IQA dataset/CSIQ/',
    'live': '/mnt/g/datasets/IQA dataset/LIVE/',
    'tid2013': '/mnt/g/datasets/IQA dataset/TID2013/',
    'koniq-10k': '/mnt/g/datasets/IQA dataset/KONIQ-10k/',
    'kadid-10k': '/mnt/g/datasets/IQA dataset/KADID-10k/',
    'pipal': '/mnt/g/datasets/IQA dataset/PIPAL/',
    'spaq': '/mnt/g/datasets/IQA dataset/SPAQ/',
    'agiqa-3k': '/mnt/g/datasets/IQA dataset/AGIQA-3K/'
    } if os.name == 'posix' else {
    'csiq': 'G:/datasets/IQA dataset/CSIQ/',
    'live': 'G:/datasets/IQA dataset/LIVE/',
    'tid2013': 'G:/datasets/IQA dataset/TID2013/',
    'koniq-10k': 'G:/datasets/IQA dataset/KONIQ-10k/',
    'kadid-10k': 'G:/datasets/IQA dataset/KADID-10k/',
    'pipal': 'G:/datasets/IQA dataset/PIPAL/',
    'spaq': 'G:/datasets/IQA dataset/SPAQ/',
    'agiqa-3k': 'G:/datasets/IQA dataset/AGIQA-3K/'
    }

img_num = {
    'live': list(range(0, 29)),
    'csiq': list(range(0, 30)),
    'tid2013': list(range(0, 25)),
    'koniq-10k': list(range(0, 10073)),
    'kadid-10k': list(range(0, 10125)),
    'pipal': list(range(0, 23200)),
    'spaq': list(range(0, 11125)),
    'agiqa-3k': list(range(0, 2982)),
    }

max_limit = {
    'csiq': 1,
    'live': 100,
    'tid2013': 9,
    'koniq-10k': 100,
    'kadid-10k': 5,
    'pipal': 2000,
    'spaq': 100,
    'agiqa-3k': 5,
    }

class DataLoader(object):
    """Dataset class for IQA databases"""
    def __init__(self, dataset, path, img_indx, patch_num=1, batch_size=1):
        
        self.batch_size = batch_size

        transforms = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]) 

        if dataset == 'csiq':
            self.data = folders.CSIQFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'live':
            self.data = folders.LIVEFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'koniq-10k':
            self.data = folders.Koniq_10kFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'kadid-10k':
            self.data = folders.Kadid_10kFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'pipal':
            self.data = folders.PIPALFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'spaq':
            self.data = folders.SPAQFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'agiqa-3k':
            self.data = folders.AGIQA3KFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def __call__(self):
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)
        return dataloader

def normalize_X(X):
    X = (X - X.min()) / (X.max() - X.min())
    if X.min() < 0 or X.max() > 1:
        X = torch.clamp(X, 0, 1)
    return X

# y 的 范围由数据集的mos决定
def normalize_y(y, datasets):
    normalized_y = y / max_limit[datasets]  # range into 0-1
    if datasets in ['csiq', 'live']:
        normalized_y = 1 - normalized_y  # negation
    return normalized_y