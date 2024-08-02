import os
import sys
import torch
import pyiqa
import random   
import argparse
import warnings
import torchvision
import numpy as np
from tqdm import tqdm
from curd import calculate_sp, prediction, expand
from data_loader import DataLoader, normalize_Mssim, normalize_mos, dataset_info


class IQA_framework():
    def __init__(self, iqa_net):
        super(IQA_framework, self).__init__()
        # load iqa_network
        self.func = iqa_net
        # Load VGG model
        self.net = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).cuda().features.eval()
        # Feature Layers ID
        self.convlayer_id = [0, 2, 5, 7, 10]
        # Sample Rate
        self.sr = np.array([64, 128, 256, 512, 512])
        # 特征图用
        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor()])

        self.to_pil = torchvision.transforms.ToPILImage()
        
    def extractFeature(self, img):
        img = torch.as_tensor(img).cuda()
        feat_map = [img]
        cnt = 0
        for i, layer in enumerate(self.net.children()):
            img = layer(img)
            if i in self.convlayer_id:
                img0 = img
                for j in range(img0.shape[1]):
                    if j % self.sr[cnt] == 0:
                        random_channels = [random.randint(0, img0.size(1) - 1) for _ in range(3)] # 生成随机的三个通道索引（0到num_channels之间，包括0和num_channels）
                        temp = torch.cat([torch.as_tensor(self.transform(self.to_pil(img0[:, c,:,:]))).unsqueeze(1) for c in random_channels], dim=1).cuda()
                        feat_map.append(temp)
                cnt = cnt + 1
        return feat_map

    def origin_framework(self, img):
        img = torch.as_tensor(img).cuda()
        pred_scores = []
        for _ in range(10):
            pred = self.func(img)
            pred_scores.append(float(pred.item()))
        return np.mean(pred_scores)

    def multiscale_framework(self, img):
        feat_map = self.extractFeature(img) # Extract feature map
        pred_scores = []
        layer_scores = []
        for feat in feat_map:
            for _ in range(10):
                pred = self.func(feat)
                pred_scores.append(float(pred.item()))
            score = np.mean(pred_scores)
            layer_scores.append(score)
        return layer_scores

def load_metrics_pyiqa(method):
    if pyiqa:
        iqa_net = pyiqa.create_metric(method, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if iqa_net.lower_better:
            print(f'The lower value of the metric {method} is better.')
        else:
            print(f'The higher value of the metric {method} is better.')
        return iqa_net

def load_maniqa(premodel):
    if './src/maniqa' not in sys.path:
        sys.path.insert(0, './src/maniqa')
        from config import Config
        from models.maniqa import MANIQA

    model_path = f'/src/maniqa/sota_ckpt_reproduce/ckpt_{premodel}.pt'
    iqa_net = MANIQA(embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8, img_size=224, window_size=4, depths=[2,2], num_heads=[4,4], num_tab=2, scale=0.8)
    iqa_net = iqa_net.load_state_dict(torch.load(model_path)).cuda()
    iqa_net.eval()

    return iqa_net

def load_clipiqa_plus():
    pass

def main(config):
    # 根据方法创建提升框架
    if config.pyiqa:
        print(pyiqa.list_models())
        iqa_net = load_metrics_pyiqa(config.method)
    elif config.method == 'maniqa':
        iqa_net = load_maniqa(config.premodel)
    elif config.method == 'clipiqa+':
        iqa_net = load_clipiqa_plus(config.premodel)
    else:
        pass

    framework = IQA_framework(iqa_net)
        
    # 读入测试数据集 (img + mos)
    dataLoader = DataLoader(config.dataset, dataset_info['folder_path'][config.dataset], dataset_info['img_num'][config.dataset], patch_size = 224, patch_num = 1, istrain=False)
    data = dataLoader.get_data()

    if config.multiscale:
        # 通过多尺度提升框架，获取 mat = Mssim + mos 矩阵
        mat = []
        for img, label in tqdm(data):
            layer_scores = framework.multiscale_framework(img)
            labels = np.array([float(label.numpy())])
            mat.append(np.hstack((layer_scores, labels)))
        mat = np.array(mat)

        if config.norm_mmsim:
            Mssim, mos = expand(normalize_Mssim(mat[:,:-1], config.premodel), 'mode2'), normalize_mos(mat[:,-1], config.dataset)[:, np.newaxis]
        else:
            Mssim, mos = expand(mat[:,:-1], 'mode2'), normalize_mos(mat[:,-1], config.dataset)[:, np.newaxis]


        if config.index is None and config.beta is None:
            # 写入 dataset.txt 文件
            print('Output the layerscores and mos...')
            
            if not os.path.exists(f'./outputs/{config.method}/multiscale outputs/'):
                os.makedirs(f'./outputs/{config.method}/multiscale outputs/')

            np.savetxt(f"./outputs/{config.method}/multiscale outputs/{config.dataset}.txt", mat, fmt='%f', delimiter='\t')

        else:
            # 多元回归
            print('Input index and beta, Evalue by the CURD...')
           

            yhat = prediction(Mssim, config.beta, config.index)

            plcc, srcc = calculate_sp(mos.squeeze(), yhat.squeeze())
            print(f'Testing PLCC {plcc},\tSRCC {srcc}.')
    else:
        # 原始的 method
        pred_scores, gt_scores = [], []
        for img, label in tqdm(data):
            score = framework.origin_framework(img)

            pred_scores.append(float(score.item()))
            gt_scores = gt_scores + label.tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)

        plcc, srcc = calculate_sp(pred_scores, gt_scores)
        print(f'Testing PLCC {plcc},\tSRCC {srcc}')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--multiscale', action='store_true', help='The flag of using multiscale framework')
    parser.add_argument('--pyiqa', action='store_true', help='The flag of using pyiqa package')

    parser.add_argument('--method', dest='method', type=str, required=True, default='dbcnn', help='Support methods: clipiqa|clipiqa+|ilniqe|wadiqam_nr|dbcnn|paq2piq|hyperiqa|tres|tres-flive|tres-koniq|maniqa|maniqa-kadid|maniqa-koniq|maniqa-pipal')
    parser.add_argument('--premodel', dest='premodel', type=str, default=None, help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--norm_mssim', action='store_true', help='The flag of mssim normalization, when the output ranges of the model pretrained by different datasets are diverse')
    
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, choices=['koniq-10k', 'live', 'csiq', 'tid2013'], default='tid2013', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--index', dest='index', nargs='+', type=int, default=None, help='List of index values')
    parser.add_argument('--beta', dest='beta', nargs='+', type=float, default=None, help='List of beta values')
    config = parser.parse_args()

    print(f'Using multiscale framework:{config.multiscale}\n Using pyiqa package:{config.pyiqa}\n Pretrained-model without normalization:{config.premodel}\n Test method:{config.method}\n Test dataset:{config.dataset}\n Prediction mode:{not (config.index is None) and not (config.beta is None)}\t')

    main(config)
