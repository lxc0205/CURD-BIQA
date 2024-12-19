import os
import torch
import folders
import torchvision
import numpy as np

folder_path ={
    'csiq': '/mnt/g/datasets/IQA dataset/CSIQ/',
    'live': '/mnt/g/datasets/IQA dataset/LIVE/',
    'tid2013': '/mnt/g/datasets/IQA dataset/TID2013/',
    'koniq-10k': '/mnt/g/datasets/IQA dataset/KONIQ-10k/',
    } if os.name == 'posix' else {
    'csiq': 'G:/datasets/IQA dataset/CSIQ/',
    'live': 'G:/datasets/IQA dataset/LIVE/',
    'tid2013': 'G:/datasets/IQA dataset/TID2013/',
    'koniq-10k': 'G:/datasets/IQA dataset/KONIQ-10k/',
    }

img_num = {
    'live': list(range(0, 29)),
    'csiq': list(range(0, 30)),
    'tid2013': list(range(0, 25)),
    'koniq-10k': list(range(0, 10073)),
    }

max_limit = {
    'csiq': 1,
    'live': 100,
    'tid2013': 9,
    'koniq-10k': 100, 
    }

class DataLoader(object):
    """Dataset class for IQA databases"""
    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True, transform_mode='default'):
        
        self.batch_size = batch_size
        self.istrain = istrain

        # Train transforms
        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.RandomHorizontalFlip(p = 0.7),  
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        # Test transforms
        else:
            if transform_mode == 'pyiqa':
                transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            elif transform_mode == 'maniqa':
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224, 224)), # maniqa-csiq,live,tid2013
                    # torchvision.transforms.Resize((512, 384)),
                    # torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(), # maniqa-csiq,live,tid2013
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # maniqa-csiq,live,tid2013
                    # torchvision.transforms.Normalize(mean=(0.500, 0.500, 0.500), std=(0.500, 0.500, 0.500)) # pyiqa maniqa orignal normalize, get from MANIQA-master
                ])
            elif transform_mode == 'default':
                transforms = torchvision.transforms.Compose([
                    # torchvision.transforms.Resize((224, 224)), # maniqa-csiq,live,tid2013
                    # torchvision.transforms.Resize((512, 384)),
                    # torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(), # maniqa-csiq,live,tid2013
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # maniqa-csiq,live,tid2013
                    # torchvision.transforms.Normalize(mean=(0.500, 0.500, 0.500), std=(0.500, 0.500, 0.500)) # pyiqa maniqa orignal normalize, get from MANIQA-master
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

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader

def load_ssim_mos(file_path):
    data = np.loadtxt(file_path)
    ssim, mos = data[:,:-1], data[:,-1]

    dataset = file_path.split('/')[-1][:-4]
    mos = normalize_mos(mos, dataset)
    mos = mos[:, np.newaxis]

    return ssim, mos

def norm_ssim(ssim, norm_range=None):
    if norm_range is not None:
        ssim = normalize_Mssim(ssim, range=norm_range)
    return ssim

def normalize_Mssim(Mssim, range):
    return np.clip(Mssim, 0, range) / range # range into 0-1
 
def normalize_mos(scores, datasets):
    normalized_scores = np.array(scores) / max_limit[datasets] #range into 0-1
    if datasets in ['csiq', 'live']:
        normalized_scores = 1 - normalized_scores # negation
    return normalized_scores