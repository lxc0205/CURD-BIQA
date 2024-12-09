import os
import sys
import torch
import pyiqa
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from IQA_framework import IQA_framework
from curd import calculate_sp, prediction, expand, beta_index_to_function
from data_loader import DataLoader, normalize_Mssim, normalize_mos, folder_path, img_num

def load_metrics_pyiqa(method):
        print(pyiqa.list_models())
        iqa_net = pyiqa.create_metric(method, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        flag = 'lower' if iqa_net.lower_better else 'higher'
        print(f'The {flag} value of the metric {method} is better.')
        def scaled_iqa_net(img, scale = 100):
            with torch.no_grad():
                score = iqa_net(img)
                return score * scale
        return scaled_iqa_net

def load_maniqa(ckpt_path):
    if './src/maniqa' not in sys.path:
        sys.path.insert(0, './src/maniqa')
        from config import Config
        from models.maniqa import MANIQA
    iqa_net = MANIQA(embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8, img_size=224, window_size=4, depths=[2,2], num_heads=[4,4], num_tab=2, scale=0.8)
    iqa_net.load_state_dict(torch.load(ckpt_path))
    iqa_net.cuda()
    iqa_net.eval()
    return iqa_net

def main(config): 
    # choose the BIQA model
    if config.pyiqa:
        iqa_net = load_metrics_pyiqa(config.method)
    elif config.method == 'maniqa':
        iqa_net = load_maniqa(config.ckpt)
    else:
        print('The method is not supported.')
    
    # create the enhancing framework
    framework = IQA_framework(iqa_net)

    # dataLoader (img + mos)
    if config.pyiqa:
        dataLoader = DataLoader(config.dataset, folder_path[config.dataset], img_num[config.dataset], patch_size = 224, patch_num = 1, istrain=False, transform_mode = 'pyiqa')
    else:
        dataLoader = DataLoader(config.dataset, folder_path[config.dataset], img_num[config.dataset], patch_size = 224, patch_num = 1, istrain=False, transform_mode = 'maniqa')
    data = dataLoader.get_data()

    if config.multiscale:
        # mutiscale framework
        mat = []
        for img, label in tqdm(data):
            layer_scores = framework.multiscale_framework(img)
            mat.append(np.hstack((layer_scores, np.array([float(label.numpy())]))))
        mat = np.array(mat)
        Mssim, mos = mat[:, :-1], mat[:, -1]

        if config.index is None and config.beta is None:
            # save to files
            print('Output the layerscores and mos...')
            multiscale_output_Path = f'./outputs/{config.method}/multiscale outputs/'
            if not os.path.exists(multiscale_output_Path): os.makedirs(multiscale_output_Path)
            np.savetxt(multiscale_output_Path + f"{config.dataset}.txt", mat, fmt='%f', delimiter='\t')

        else:
            # test models by beta and index
            print('Input index and beta, Evalue by the CURD-IQA-enhanced method...')
            function, function_latex = beta_index_to_function(config.index, config.beta)
            print(f"final function: {function}, \n{function_latex}")
            Mssim = (
            expand(normalize_Mssim(Mssim, config.norm_R))
            if config.norm_R is not None else expand(Mssim))
            mos = normalize_mos(mos, config.dataset)[:, np.newaxis]
            yhat = prediction(Mssim, config.beta, config.index)
            plcc, srcc = calculate_sp(mos.squeeze(), yhat.squeeze())
            print(f'Testing PLCC {plcc},\tSRCC {srcc}.')

    else:
        # orignal framework
        scores, labels = [], []
        for img, label in tqdm(data):
            score = framework.origin_framework(img)
            scores.append(float(score.item()))
            labels = labels + label.tolist()
        plcc, srcc = calculate_sp(np.array(scores), np.array(labels))
        print(f'Testing PLCC {plcc},\tSRCC {srcc}.')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--multiscale', action='store_true', help='The flag of using multiscale framework.')
    parser.add_argument('--method', dest='method', type=str, required=True, default='dbcnn', help='Support methods: clipiqa+|wadiqam_nr|dbcnn|paq2piq|hyperiqa|tres|tres-flive|tres-koniq|maniqa|maniqa-kadid|maniqa-koniq|maniqa-pipal')
    parser.add_argument('--pyiqa', action='store_true', help='The flag of using pyiqa package.')
    parser.add_argument('--ckpt', dest='ckpt', type=str, default=None, help='The checkpoint path.')
    parser.add_argument('--norm_R', dest='norm_R', type=int, default=None, help='The range of mssim normalization.')
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, choices=['csiq', 'live', 'tid2013', 'koniq-10k'], default='csiq', help='Support datasets: csiq|live|tid2013|koniq-10k.')
    parser.add_argument('--index', dest='index', nargs='+', type=int, default=None, help='List of index values.')
    parser.add_argument('--beta', dest='beta', nargs='+', type=float, default=None, help='List of beta values.')
    config = parser.parse_args()

    # print configs
    config_dict = vars(config)
    print("Configs:")
    for key, value in sorted(config_dict.items()):
        print(f"{key.replace('_', ' ').title()}: {value}")

    main(config)
