import torch
import folders
import torchvision

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):
        
        self.batch_size = batch_size
        self.istrain = istrain

        if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'tid2013'):
            # Train transforms
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
            # Test transforms
            else:
                # transforms = torchvision.transforms.Compose([
                #     torchvision.transforms.ToTensor(),
                #     torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                #                                      std=(0.229, 0.224, 0.225))
                #                                      ])
                transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        elif dataset == 'koniq-10k':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                # transforms = torchvision.transforms.Compose([
                #     torchvision.transforms.Resize((512, 384)),
                #     torchvision.transforms.RandomCrop(size=patch_size),
                #     torchvision.transforms.ToTensor(),
                #     torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                #                                      std=(0.229, 0.224, 0.225))])
                transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


        if dataset == 'live':
            self.data = folders.LIVEFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'csiq':
            self.data = folders.CSIQFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'koniq-10k':
            self.data = folders.Koniq_10kFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader

# 
folder_path = {
    'live': 'G:/datasets/IQA dataset/LIVE/',
    'csiq': 'G:/datasets/IQA dataset/CSIQ/',
    'tid2013': 'G:/datasets/IQA dataset/TID2013/',
    'koniq-10k': 'G:/datasets/IQA dataset/KONIQ-10k/',
}

img_num = {
    'live': list(range(0, 29)),
    'csiq': list(range(0, 30)),
    'tid2013': list(range(0, 25)),
    'koniq-10k': list(range(0, 10073)),
}

max_limit_value = {
    'csiq': 1,
    'live': 100,
    'tid2013': 9,
    'koniq-10k': 100,
}

import numpy as np

def normalize_Mssim(Mssim, datasets):
    return np.clip(Mssim, 0, max_limit_value[datasets]) / max_limit_value[datasets]
 
def normalize_mos(scores, datasets, new_min=0, new_max=1):
    old_min = 0 
    old_max = max_limit_value[datasets]
    normalized_scores = [((new_max - new_min) * (score - old_min) / (old_max - old_min) + new_min) for score in scores]
    if datasets in ['csiq', 'live']:
        normalized_scores = [1 - score for score in normalized_scores]
    return np.array(normalized_scores)
    
def loadMssimMos_single(file_path):    
    
    mat = np.loadtxt(file_path)
    Mssim, mos = mat[:,:-1], mat[:,-1]
    return Mssim, mos

def loadMssimMos(file_list, sep = -1):
    for index, path in enumerate(file_list):
        Mssim0, mos0 = MssimMosPath(path)
        if sep != -1:
            Mssim0, mos0 = Mssim0[:sep], mos0[:sep]
        
        if index == 0:
            Mssim, Mos = Mssim0, mos0
        else:    
            Mssim, Mos = np.concatenate((Mssim, Mssim0), axis=0), np.concatenate((Mos, mos0), axis=0)

    return Mssim, Mos

# 存在问题 这里hyperIQA有很多pretrain model 建议手动设置 pretrained model
# def MssimMosPath_for_HyperIQA(file_path):
#     parts = file_path.split('_')
#     dataset, pretrained_dataset = parts[0], parts[1][:-4]

#     Mssim, mos = loadMssimMos_single("./outputs/hyperIQA outputs/" + file_path)
#     Mssim, mos = expand2(normalize_Mssim(Mssim, pretrained_dataset)), normalize_mos(mos, dataset)[:, np.newaxis] # 归一化到0-1
#     return Mssim, mos

def MssimMosPath(file_path):
    parts = file_path.split('/')
    dataset = parts[-1][:-4] # 去除.txt

    Mssim, mos = loadMssimMos_single(file_path)
    Mssim, mos = Mssim, normalize_mos(mos, dataset)[:, np.newaxis] # 归一化到0-1
    return Mssim, mos