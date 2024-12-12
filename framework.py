import os
import sys
import json
import torch
import pyiqa
import random
import argparse
import warnings
import torchvision  
import numpy as np
from tqdm import tqdm
from datetime import datetime
from curd import CURD, calculate_sp, regression, prediction, expand, sort, beta_index_to_function
from dataLoader import DataLoader, loadMssimMos, normalize_Mssim, normalize_mos, folder_path, img_num


class IQAFramework():
    def __init__(self, iqa_net):
        super(IQAFramework, self).__init__()
        # load iqa_network
        self.function = iqa_net
        # Load VGG16 model
        self.net = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).cuda().features.eval()
        # Feature Layers ID
        # self.convlayer_id = [0, 2, 5, 7, 10] # original layers
        # self.convlayer_id = [1, 3, 5, 7, 9, 11, 13, 16, 18, 20, 23, 25, 27, 30]
        # self.convlayer_id = [1, 5, 9, 13, 18, 23, 27]
        # self.convlayer_id = [1, 9, 18, 27]
        self.convlayer_id = [1, 5, 9, 18, 27]
        # Sample Rate
        self.sr = np.array([64, 128, 256, 512, 512])
        # Transform for feature maps
        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor()])
        self.to_pil = torchvision.transforms.ToPILImage()
        
    def extract_feature(self, original_image):
        image = torch.as_tensor(original_image).cuda()
        feature_maps = [original_image]
        cnt = 0
        for i, layer in enumerate(self.net.children()):
            image = layer(image)
            if i in self.convlayer_id:
                feature = image
                for j in range(feature.shape[1]):
                    if j % self.sr[cnt] == 0:
                        random_channels = [random.randint(0, feature.size(1) - 1) for _ in range(3)] # 生成随机的三个通道索引
                        temp = torch.cat([torch.as_tensor(self.transform(self.to_pil(feature[:, c,:,:]))).unsqueeze(1) for c in random_channels], dim=1).cuda()
                        feature_maps.append(temp)
                cnt = cnt + 1
        return feature_maps

    def origin_framework(self, image, mean=False):
        image = torch.as_tensor(image).cuda()
        if mean:
            pred_scores = []
            for _ in range(10):
                pred = self.function(image)
                pred_scores.append(float(pred.item()))
            return np.mean(pred_scores)
        else:
            return self.function(image)

    def multiscale_framework(self, image, mean=False):
        feature_maps = self.extractFeature(image) # Extract feature map
        layer_scores = []
        for feature in feature_maps:
            if mean:
                pred_scores = []
                for _ in range(10):
                    pred = self.function(feature)
                    pred_scores.append(float(pred.item()))
                score = np.mean(pred_scores)
            else:
                pred = self.function(feature)
                score = float(pred.item())
            layer_scores.append(score)
        return layer_scores

def load_metrics_pyiqa(method):
        print(pyiqa.list_models())
        iqa_net = pyiqa.create_metric(method, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        flag = 'lower' if iqa_net.lower_better else 'higher'
        print(f'The {flag} value of the metric {method} is better.')
        def scaled_iqa_net(image, scale = 100):
            with torch.no_grad():
                score = iqa_net(image)
                return score * scale
        return scaled_iqa_net

def load_maniqa(ckpt_path):
    if './src/maniqa' not in sys.path:
        sys.path.insert(0, './src/maniqa')
        from config import Config
        from models.maniqa import MANIQA
    iqa_net = MANIQA(embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8, img_size=224, window_size=4, depths=[2,2], num_heads=[4,4], num_tab=2, scale=0.8)
    iqa_net.load_state_dict(torch.load(ckpt_path))
    iqa_net.cuda().eval()
    # iqa_net.eval()
    return iqa_net

def curd_process(config):
    method, input_files, output_file, save_num, norm_Rs, remove_cache = config['method'], config['input_files'], config['output_file'], config['save_num'], config['norm_Rs'], config['remove_cache']
    # Load data
    output_path = f'./outputs/{method}/'
    input_files = [output_path + 'multiscale outputs/' + item for item in input_files]
    output_file = output_path + 'curd outputs/' + output_file
    if not os.path.exists(output_path + 'curd outputs/'):
        os.makedirs(output_path + 'curd outputs/')

    Mssim, mos = loadMssimMos(input_files)

    temp_file = output_path + 'curd_temp.txt'
    curd = CURD(Mssim, mos.squeeze(1), temp_file)
    if os.path.exists(temp_file):
        sorted_matrix = np.loadtxt(temp_file)
    else:
        sorted_matrix = curd.process(save_num)
        
    # Perform regression evaluation and save data
    regressing(input_files, output_file, curd.NO, sorted_matrix, save_num, norm_Rs)
    def regressing(input_files, output_file, no, sorted_matrix, save_num, norm_R_set):
        # BL = np.array([0.9680,0.9610,0.9830,0.9820,0.9430,0.9370,0.9460,0.9300])
        BL_plcc, BL_srcc = np.array([0.9680,0.9830,0.9430,0]), np.array([0.9610,0.9820,0.9370,0])
        ssim_list, mos_list = [], []
        for id, dataset in enumerate(input_files):
            ssim_temp, mos_temp = loadMssimMos({dataset}, [norm_R_set[id]])
            ssim_list.append(expand(ssim_temp))
            mos_list.append(mos_temp)
        # matrix 结构:
        # 0 1 2 3 4 5 6    7    8 - 14    15 - 21    22 - 28    29 - 35    36 37 38 39    40 41 42 43     44
        # ----index----   sw    beta1      beta2      beta3      beta4         srcc           plcc       sum/8
        matrix = np.zeros((save_num, 2*no + 31))
        for epoch, row in tqdm(enumerate(sorted_matrix), total=len(sorted_matrix)):
            plcc_list, srcc_list, beta_mat = [0]*4, [0]*4, [[0]*7]*4
            for i in range(len(ssim_list)):
                index = row[:no].astype(int)
                beta_mat[i] = regression(ssim_list[i], mos_list[i], index)
                yhat = prediction(ssim_list[i], beta_mat[i], index)
                plcc_list[i], srcc_list[i] = calculate_sp(mos_list[i].squeeze(), yhat.squeeze())

            rounded_plcc_list = np.round(plcc_list, decimals=3)
            rounded_srcc_list = np.round(srcc_list, decimals=3)
            plcc_diff = [plcc - BL_plcc[i] for i, plcc in enumerate(rounded_plcc_list)]
            srcc_diff = [srcc - BL_srcc[i] for i, srcc in enumerate(rounded_srcc_list)]
            if all(x >= 0 for x in plcc_diff) and all(x >= 0 for x in srcc_diff):
                matrix[epoch] = np.concatenate((row[:no+1], beta_mat[0].squeeze(), beta_mat[1].squeeze(), beta_mat[2].squeeze(), beta_mat[3].squeeze(), plcc_list, srcc_list,[(sum(plcc_list)+sum(srcc_list))/8]))
        print(f'Number of regression items: {epoch}\n')
        # sort and save into a file
        matrix = sort(matrix, order="descending", row = 44)[:save_num, :]
        np.savetxt(output_file, matrix, fmt=['%d']*no + ['%f']*(matrix.shape[1]-no), delimiter=' ')

        if remove_cache:
            print(f'Remove cache files...')
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print(f'The curd iqa finished!')

def method_process(configs): 
    # get configs
    mode, dataset, method, ckpt, norm_R, index, beta = configs['mode'], configs['dataset'], configs['method'], configs['ckpt'], configs['norm_R'], configs['index'], configs['beta']

    # create the enhancing framework
    if ckpt is None:
        iqa_net = load_metrics_pyiqa(method)
    elif method == 'maniqa':
        iqa_net = load_maniqa(ckpt)
    else:
        print('The method is not supported.')
        return
    framework = IQAFramework(iqa_net)

    # dataLoader (img + mos)
    dataLoader = DataLoader(dataset, folder_path[dataset], img_num[dataset], patch_size = 224, patch_num = 1, istrain=False, transform_mode = ('pyiqa' if pyiqa else 'maniqa'))
    data = dataLoader.get_data()

    if mode == 'original': # orignal framework
        scores, labels = [], []
        for image, label in tqdm(data):
            score = framework.origin_framework(image)
            scores.append(float(score.item()))
            labels = labels + label.tolist()
        plcc, srcc = calculate_sp(np.array(scores), np.array(labels))
        print(f'Testing PLCC {plcc},\tSRCC {srcc}.')

    if mode == 'multiscale' or mode == 'enhanced': # mutiscale framework
        mat = []
        for img, label in tqdm(data):
            layer_scores = framework.multiscale_framework(img)
            mat.append(np.hstack((layer_scores, np.array([float(label.numpy())]))))
        mat = np.array(mat)
        Mssim, mos = mat[:, :-1], mat[:, -1]
        print(method)

    if mode == 'multiscale': # mutiscale framework
        # save to files
        print('Output the layerscores and mos...')
        multiscale_output_Path = f'./outputs/{method}/multiscale outputs/'
        if not os.path.exists(multiscale_output_Path): os.makedirs(multiscale_output_Path)
        np.savetxt(multiscale_output_Path + f"{dataset}.txt", mat, fmt='%f', delimiter='\t')
    
    if mode == 'enhanced':
        # test models by beta and index
        print('Input index and beta, evalue by the CURD-IQA-enhanced method...')
        # beta_index_to_function(index, beta)
        Mssim = (expand(normalize_Mssim(Mssim, norm_R)) if norm_R is not None else expand(Mssim))
        mos = normalize_mos(mos, dataset)[:, np.newaxis]
        yhat = prediction(Mssim, beta, index)
        plcc, srcc = calculate_sp(mos.squeeze(), yhat.squeeze())
        print(f'Testing PLCC {plcc},\tSRCC {srcc}.')


if __name__ == '__main__':
    # ingore warnings
    warnings.filterwarnings('ignore')

    # load json file as configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()
    with open('./configs/' + args.config + '.json', 'r') as file:
        configs = json.load(file)

    # show configs
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("configs:")
    for key, value in sorted(configs.items()):
        print(f"{key.replace('_', ' ').title()}: {value}")

    # main process
    if configs['mode'] == 'original' or configs['mode'] == 'multiscale' or configs['mode'] == 'enhanced':
        method_process(configs)
    elif configs['mode'] == 'curd':
        curd_process(configs)
    else:
        print('The mode is not supported.')
        exit(0)