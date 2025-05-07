import os
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from framework import feature_framework
from method_loader import load_methods
from data_utils import save_parameter, load_parameter, save_matrix, load_matrix, create_directory
from curd import CURD, calculate_sp, regression, prediction, expand, sort, beta_index_to_function
from data.dataLoader import DataLoader, load_ssim_mos, norm_ssim, normalize_Mssim, normalize_mos, folder_path, img_num

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

def origin_process(configs):
    # load iqa method
    iqa_net, transform_mode = load_methods(configs['method'], configs['origin']['ckpt'])

    # load dataset: img + mos
    dataLoader = DataLoader(configs['origin']['dataset'], folder_path[configs['origin']['dataset']], img_num[configs['origin']['dataset']], patch_size = 224, patch_num = 1, istrain=False, transform_mode = transform_mode)
    data = dataLoader.get_data()
    
    # create framework
    frame = feature_framework(iqa_net)

    # evaluate the model
    scores, labels = [], []
    for image, label in tqdm(data):
        score = frame.origin(image)
        scores.append(float(score.item()))
        labels = labels + label.tolist()
    plcc, srcc = calculate_sp(np.array(scores), np.array(labels))
    print(f'Testing plcc {plcc},\tsrcc {srcc}.')

def curd_process(configs):
    for dataset_id, dataset in enumerate(configs['curd']['datasets']):
        # load iqa method
        iqa_net, transform_mode = load_methods(configs['method'], configs['curd']['ckpts'][dataset_id]) 

        # load dataset: img + mos
        dataLoader = DataLoader(dataset, folder_path[dataset], img_num[dataset], patch_size = 224, patch_num = 1, istrain=False, transform_mode = transform_mode)
        data = dataLoader.get_data()

        # create framework
        frame = feature_framework(iqa_net)

        # extract multiscale features
            # matrix 0 1 2 3 4 5 6    7    8 - 14    15 - 21    22 - 28    29 - 35    36 37 38 39    40 41 42 43     44
            #        ----index----   sw   betas 1    betas 2    betas 3    betas 4       srcc           plcc       sum/8
        matrix = []
        for img, label in tqdm(data):
            layer_scores = frame.multiscale(img)
            matrix.append(np.hstack((layer_scores, label.numpy().astype(float))))
        # save matrix
        create_directory('./outputs/' + configs['method'] + '/multiscale outputs/')
        save_matrix(torch.tensor(np.array(matrix), dtype=torch.float32), './outputs/' + configs['method'] + '/multiscale outputs/' + dataset + '.pt')

    # file paths
    create_directory('./outputs/' + configs['method'] + '/curd outputs/')
    input_files = ['./outputs/' + configs['method'] + '/multiscale outputs/' + item for item in configs['input_files']]
    output_file = './outputs/' + configs['method'] + '/curd outputs/' + configs['output_file']
    temp_file = './outputs/' + configs['method'] + '/curd outputs/' + 'curd_temp.txt'

    # load ssim and mos
    ssims, moss = [], []
    ssims_for_curd = []
    for id, dataset in enumerate(input_files):
        ssim, mos = load_ssim_mos(dataset)
        ssims_for_curd.append(ssim)
        ssim = norm_ssim(ssim, configs['norm_Rs'][id])
        ssims.append(expand(ssim))
        moss.append(mos)
    
    mssim_concat, mos_concat = np.concatenate(ssims_for_curd, axis=0), np.concatenate(moss, axis=0)

    curd = CURD(mssim_concat, mos_concat.squeeze(1), temp_file)
    curd_outputs = np.loadtxt(temp_file) if os.path.exists(temp_file) else curd.process(configs['save_num'])
        
    # perform regression evaluation and save data
    baseline_plcc, baseline_srcc = np.array([0.968,0.983,0.943,0]), np.array([0.961,0.982,0.937,0]) # 0, 0 -> 0.946, 0.9300
        
    no = curd.NO
    matrix = np.zeros((configs['save_num'], 2*no + 31))
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
    matrix = sort(matrix, order="descending", row = 44)[:configs['save_num'], :]
    np.savetxt(output_file, matrix, fmt=['%d']*no + ['%f']*(matrix.shape[1]-no), delimiter=' ')

    if configs['rm_cache']:
        print('remove cache files...')
        if os.path.exists(temp_file):
            os.remove(temp_file)
    print(f'The curd iqa finished!')

def enhanced_process(configs):
    if configs['enhanced']['dataloading']:
        matrix = load_matrix('./outputs/' + configs['method'] + '/multiscale outputs/'+ configs['enhanced']['dataset'] + '.pt')
        matrix = np.array(matrix)
    else:
        # load iqa method
        iqa_net, transform_mode = load_methods(configs['method'], configs['enhanced']['ckpt'])

        # load dataset: img + mos
        dataLoader = DataLoader(configs['enhanced']['dataset'], folder_path[configs['enhanced']['dataset']], img_num[configs['enhanced']['dataset']], patch_size = 224, patch_num = 1, istrain=False, transform_mode = transform_mode)
        data = dataLoader.get_data()

        # create framework
        frame = feature_framework(iqa_net)

        # evaluate the model
        matrix = []
        for img, label in tqdm(    data = data.numpy()):
            layer_scores = frame.multiscale(img)
            matrix.append(np.hstack((layer_scores, label.numpy().astype(float))))
        matrix = np.array(matrix)

    # load parameters
    index, beta = load_parameter(configs['enhanced']['nonliear_model_path'])
    beta_index_to_function(index, beta)

    Mssim, mos = matrix[:, :-1], matrix[:, -1]
    Mssim = (expand(normalize_Mssim(Mssim, configs['norm_R'])) if configs['norm_R'] is not None else expand(Mssim))
    mos = normalize_mos(mos, configs['enhanced']['dataset'])[:, np.newaxis]
    yhat = prediction(Mssim, beta, index)
    plcc, srcc = calculate_sp(mos.squeeze(), yhat.squeeze())
    print(f'Testing plcc {plcc},\tsrcc {srcc}.')

    dataset = configs['enhanced']['dataset']
    output_file = f'enhanced_results_{dataset}.txt'
    np.savetxt(output_file, np.column_stack((mos.squeeze(), yhat.squeeze())), fmt='%f', delimiter='\t')
    print(f'Results saved to {output_file}.')

if __name__ == '__main__':
    '''
        mode: `
            origin    : original version of the method.
            curd      : curd training framework.
            enhanced  : enhanced version of the method.
    '''
    # load yaml file as configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        configs = OmegaConf.load(file)

    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("configs:")
    for key, value in sorted(configs.items()):
        print(f"{key.replace('_', ' ').title()}: {value}")

    if configs['mode'] == 'origin':
        origin_process(configs)
    elif configs['mode'] == 'curd':
        curd_process(configs)
    elif configs['mode'] == 'enhanced':
        enhanced_process(configs)
    else:
        print('The mode is not supported.')
        exit(0)
    