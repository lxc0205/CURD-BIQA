import torch
import folders
import torchvision
import numpy as np

dataset_info = {
    'folder_path': {
        'live': 'G:/datasets/IQA dataset/LIVE/',
        'csiq': 'G:/datasets/IQA dataset/CSIQ/',
        'tid2013': 'G:/datasets/IQA dataset/TID2013/',
        'koniq-10k': 'G:/datasets/IQA dataset/KONIQ-10k/',
    },
    'img_num': {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'koniq-10k': list(range(0, 10073)),
    },
    'max_limit_value': {
        'csiq': 1,
        'live': 100,
        'tid2013': 9,
        'koniq-10k': 100,
    }
}

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):
        
        self.batch_size = batch_size
        self.istrain = istrain

        # Train transforms
        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
            ])
        # Test transforms
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                                                    ])
            # transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
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
        '''
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
        '''

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader

def normalize_Mssim(Mssim, datasets):
    return np.clip(Mssim, 0, dataset_info['max_limit_value'][datasets]) / dataset_info['max_limit_value'][datasets]
 
def normalize_mos(scores, datasets, new_min=0, new_max=1):
    old_min = 0 
    old_max = dataset_info['max_limit_value'][datasets]
    normalized_scores = [((new_max - new_min) * (score - old_min) / (old_max - old_min) + new_min) for score in scores]
    if datasets in ['csiq', 'live']:
        normalized_scores = [1 - score for score in normalized_scores]
    return np.array(normalized_scores)
   
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

def MssimMosPath(file_path):
    data = np.loadtxt(file_path)
    Mssim, mos = data[:,:-1], data[:,-1]

    return normalize_Mssim(Mssim, datasets='csiq'), normalize_mos(mos, file_path.split('/')[-1][:-4])[:, np.newaxis] # 归一化到0-1