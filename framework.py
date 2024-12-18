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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
    def extract_feature(self, image):
        image = torch.as_tensor(image).cuda()
        feature_maps = [image]
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
        feature_maps = self.extract_feature(image) # Extract feature map
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
        iqa_net = pyiqa.create_metric(method, device)
        flag = 'lower' if iqa_net.lower_better else 'higher'
        print(f'The {flag} value of the metric {method} is better.')
        def scaled_iqa_net(image, scale = 100):
            with torch.no_grad():
                score = iqa_net(image)
                return score * scale
        return scaled_iqa_net

def load_maniqa(ckpt_path):
    if './src/' not in sys.path: sys.path.insert(0, './src/')
    if './src/maniqa' not in sys.path: sys.path.insert(0, './src/maniqa')
    from maniqa.models.maniqa import MANIQA
    iqa_net = MANIQA(embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8, img_size=224,
                    window_size=4, depths=[2,2], num_heads=[4,4], num_tab=2, scale=0.8)
    iqa_net.load_state_dict(torch.load(ckpt_path))
    iqa_net = iqa_net.cuda()
    iqa_net.eval()
    return iqa_net

# matrix 结构:
# 0 1 2 3 4 5 6    7    8 - 14    15 - 21    22 - 28    29 - 35    36 37 38 39    40 41 42 43     44
# ----index----   sw   betas 1    betas 2    betas 3    betas 4       srcc           plcc       sum/8
def curd_process(input_path, input_files, output_path, output_file, norm_Rs, save_num, rm_cache):
    # file paths
    input_files = [input_path + item for item in input_files]
    output_file = output_path + output_file
    temp_file = output_path + 'curd_temp.txt'

    # load ssim and mos
    Mssim, mos = loadMssimMos(input_files)
    curd = CURD(Mssim, mos.squeeze(1), temp_file)
    if os.path.exists(temp_file):
        curd_outputs = np.loadtxt(temp_file)
    else:
        curd_outputs = curd.process(save_num)
        
    # perform regression evaluation and save data
    baseline_plcc, baseline_srcc = np.array([0.968,0.983,0.943,0]), np.array([0.961,0.982,0.937,0]) # 0, 0 -> 0.946, 0.9300
    ssims, moss = [], []
    for id, dataset in enumerate(input_files):
        ssim, mos = loadMssimMos({dataset}, [norm_Rs[id]])
        ssims.append(expand(ssim))
        moss.append(mos)
        
    no = curd.NO
    matrix = np.zeros((save_num, 2*no + 31))
    for epoch, row in tqdm(enumerate(curd_outputs), total=len(curd_outputs)):
        plccs, srccs, beta_matrix = [0]*4, [0]*4, [[0]*7]*4
        for i, ssim in enumerate(ssims):
            index = row[:no].astype(int)
            beta_matrix[i] = regression(ssim, moss[i], index)
            yhat = prediction(ssim, beta_matrix[i], index)
            plccs[i], srccs[i] = calculate_sp(moss[i].squeeze(), yhat.squeeze())

        # difference_plccs = [plcc - baseline_plcc[i] for i, plcc in enumerate(np.round(plccs, decimals=3))]
        # difference_srccs = [srcc - baseline_srcc[i] for i, srcc in enumerate(np.round(srccs, decimals=3))]
        # if all(x >= 0 for x in difference_plccs) and all(x >= 0 for x in difference_srccs):
        # 0.937
        matrix[epoch] = np.concatenate((row[:no+1], beta_matrix[0].squeeze(), beta_matrix[1].squeeze(), 
                                        beta_matrix[2].squeeze(), beta_matrix[3].squeeze(), 
                                        plccs, srccs,[(sum(plccs)+sum(srccs))/8]))
    print(f'number of regression items: {epoch}\n')
    # sort and save into a file
    matrix = sort(matrix, order="descending", row = 44)[:save_num, :]
    np.savetxt(output_file, matrix, fmt=['%d']*no + ['%f']*(matrix.shape[1]-no), delimiter=' ')

    if rm_cache:
        print('remove cache files...')
        if os.path.exists(temp_file):
            os.remove(temp_file)
    print(f'The curd iqa finished!')

def method_process(mode, dataset, method, ckpt, norm_R, index, beta, output_path):
    # create the enhancing framework
    if ckpt is None:
        iqa_net = load_metrics_pyiqa(method)
        transform_mode = 'pyiqa'
    elif method == 'maniqa':
        iqa_net = load_maniqa(ckpt)
        transform_mode = 'maniqa'
    else:
        print('The method is not supported.')
        return
    framework = IQAFramework(iqa_net)

    # dataLoader (img + mos)
    dataLoader = DataLoader(dataset, folder_path[dataset], img_num[dataset], patch_size = 224, 
                            patch_num = 1, istrain=False, transform_mode = transform_mode)
    data = dataLoader.get_data()

    if mode == 'original': # orignal method
        scores, labels = [], []
        for image, label in tqdm(data):
            score = framework.origin_framework(image)
            scores.append(float(score.item()))
            labels = labels + label.tolist()
        plcc, srcc = calculate_sp(np.array(scores), np.array(labels))
        print(f'Testing PLCC {plcc},\tSRCC {srcc}.')

    if mode == 'multiscale' or mode == 'enhanced':
        matrix = []
        for img, label in tqdm(data):
            layer_scores = framework.multiscale_framework(img)
            # matrix.append(np.hstack((layer_scores, np.array([float(label.numpy())]))))
            matrix.append(np.hstack((layer_scores, label.numpy().astype(float))))
        matrix = np.array(matrix)

    if mode == 'multiscale': # mutiscale framework
        np.savetxt(output_path + dataset + '.txt', matrix, fmt='%f', delimiter='\t')
    
    if mode == 'enhanced': # enhanced method
        # beta_index_to_function(index, beta)
        Mssim, mos = matrix[:, :-1], matrix[:, -1]
        Mssim = (expand(normalize_Mssim(Mssim, norm_R)) if norm_R is not None else expand(Mssim))
        mos = normalize_mos(mos, dataset)[:, np.newaxis]
        yhat = prediction(Mssim, beta, index)
        plcc, srcc = calculate_sp(mos.squeeze(), yhat.squeeze())
        print(f'Testing PLCC {plcc}, \tSRCC {srcc}.')


if __name__ == '__main__':
    '''
        mode: 
            original  : original version of the method.
            multiscale: multiscale framework for the method. get the multiscale scores.
            curd      : curd framework. get the beta and index by curd.
            enhanced  : enhanced version of the method. test models by beta and index.
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # ingore warnings
    warnings.filterwarnings('ignore')

    # load json file as configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()
    json_path = args.json
    # json_path = './configs/temple.json'
    with open(json_path, 'r') as file:
        configs = json.load(file)

    # add input and output paths, create output folder
    if configs['mode'] == 'multiscale':
        configs['output_path'] =  './outputs/' + configs['method'] + '/multiscale outputs/'
        if not os.path.exists(configs['output_path']): 
            os.makedirs(configs['output_path'])
    if configs['mode'] == 'curd':
        configs['input_path'] = './outputs/' + configs['method'] + '/multiscale outputs/'
        configs['output_path'] = './outputs/' + configs['method'] + '/curd outputs/'
        if not os.path.exists(configs['output_path']): 
            os.makedirs(configs['output_path'])

    # show configs
    if configs['mode'] == 'original':
        used_config = ['mode', 'dataset', 'method', 'ckpt', 'norm_R']
    if configs['mode'] == 'multiscale':
        used_config = ['mode', 'dataset', 'method', 'ckpt', 'norm_R', 'output_path']
    if configs['mode'] == 'curd':
        used_config = ['mode', 'method', 'input_path', 'input_files', 'output_path', 'output_file', 'norm_Rs', 'save_num', 'rm_cache']
    if configs['mode'] == 'enhanced':
        used_config = ['mode', 'dataset', 'method', 'ckpt', 'norm_R', 'index', 'beta']
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("configs:")
    for key, value in sorted(configs.items()):
        if key in used_config:
            print(f"{key.replace('_', ' ').title()}: {value}")

    # main process
    if configs['mode'] == 'original' or configs['mode'] == 'multiscale' or configs['mode'] == 'enhanced':
        method_process( configs['mode'], configs['dataset'], configs['method'], configs['ckpt'], configs['norm_R'], configs['index'], configs['beta'], configs['output_path'])
    elif configs['mode'] == 'curd':
        curd_process(configs['input_path'], configs['input_files'], configs['output_path'], configs['output_file'], 
                      configs['norm_Rs'], configs['save_num'], configs['rm_cache'])
    else:
        print('The mode is not supported.')
        exit(0)