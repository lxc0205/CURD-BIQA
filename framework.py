import os
import torch
import pyiqa
import random   
import argparse
import warnings
import torchvision
import numpy as np
from tqdm import tqdm
from curd import calculate_sp, prediction, expand2
from data_loader import DataLoader, normalize_mos, folder_path, img_num


class IQA_framework():
    def __init__(self, method):
        super(IQA_framework, self).__init__()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.func = pyiqa.create_metric(method, device=device)
        if self.func.lower_better:
            print(f'The lower value of the metric {method} is better.')
        else:
            print(f'The higher value of the metric {method} is better.')
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
    
    def showListModels(self):
        print(pyiqa.list_models())

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
        final_score = np.mean(layer_scores)
        return layer_scores, final_score

def main(config):
    dataLoader = DataLoader(config.dataset, folder_path[config.dataset], img_num[config.dataset], patch_size = 224, patch_num = 1, istrain=False)
    data = dataLoader.get_data()

    framework = IQA_framework(config.method)
    
    if config.index is None and config.beta is None:
        print('Output the layerscores and mos...')
    else:
        print('Input index and beta, Evalue by the CURD...')
    
    if config.curd:
        # 多尺度, 写入文档
        mat = []
        for img, label in tqdm(data):
            layer_scores, _ = framework.multiscale_framework(img)
            label_array = np.array([float(label.numpy())])
            combined_scores = np.hstack((layer_scores, label_array))
            mat.append(combined_scores)
        mat = np.array(mat)

        if config.index is None and config.beta is None:
            if not os.path.exists(f'./Outputs/{config.method}/multiscale outputs/'):
                os.makedirs(f'./Outputs/{config.method}/multiscale outputs/')
            np.savetxt(f"./outputs/{config.method}/multiscale outputs/{config.dataset}.txt", mat, fmt='%f', delimiter='\t')
        else:
            index, beta = config.index, config.beta
            Mssim, mos = expand2(mat[:,:-1]), normalize_mos(mat[:,-1], config.dataset)[:, np.newaxis]
            yhat = prediction(Mssim, beta, index)
            plcc, srcc = calculate_sp(mos.squeeze(), yhat.squeeze())
            print(f'Testing PLCC {plcc},\tSRCC {srcc}.')
    else:
        # 原始IQA
        pred_scores = []
        gt_scores = []
        for img, label in tqdm(data):
            score = framework.origin_framework(img)

            pred_scores.append(float(score.item()))
            gt_scores = gt_scores + label.tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)
        plcc, srcc = calculate_sp(pred_scores, gt_scores)
        print(f'Testing PLCC {plcc},\tSRCC {srcc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='tid2013', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--method', dest='method', type=str, default='dbcnn', help='Support methods: clipiqa|clipiqa+|ilniqe|wadiqam_nr|dbcnn|paq2piq|hyperiqa|tres|tres-flive|tres-koniq|maniqa|maniqa-kadid|maniqa-koniq|maniqa-pipal')
    parser.add_argument('--index', dest='index', nargs='+', type=int, default=None, help='List of index values')
    parser.add_argument('--beta', dest='beta', nargs='+', type=float, default=None, help='List of beta values')
    parser.add_argument('--curd', action='store_true', help='The flag of using curd')
    config = parser.parse_args()
    print(f'Test method:{config.method},\tTest method:{config.dataset},\tUsing multiscale framework:{config.curd},\tPrediction mode:{not (config.index is None) and not (config.beta is None)}')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 忽略所有警告
    warnings.filterwarnings('ignore')
    main(config)
