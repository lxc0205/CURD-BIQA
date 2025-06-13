import os
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from framework import feature_framework
from method_loader import MethodLoader
from data.dataLoader import DataLoader, normalize_X, normalize_y, folder_path, img_num
from curd import CURD, calculate_sp, regression, prediction, expand, beta_index_to_function
from utils import save_parameter, load_parameter, save_matrix, load_matrix, save_logs, create_directory


def origin_process(configs):
    # load iqa method
    methodloader = MethodLoader(configs['method'], configs['pyiqa'], configs['origin']['ckpt'])
    iqa_method, transform_mode = methodloader()

    # load dataset: img + y
    dataloader = DataLoader(configs['origin']['dataset'], folder_path[configs['origin']['dataset']], img_num[configs['origin']['dataset']], transform_mode = transform_mode)
    data = dataloader.get_data()
    
    # create framework
    frame = feature_framework(iqa_method)  

    # evaluate the model
    X, y = frame.origin_loader(data)
    calculate_sp(X, y)


def curd_process(configs):
    # load multiscale score
    X_list, y_list = [], []

    for dataset_id, dataset in enumerate(configs['curd']['datasets']):
        if configs['curd']['multiscale_flag']:
            # load iqa method
            methodloader = MethodLoader(configs['method'], configs['pyiqa'], configs['curd']['ckpts'][dataset_id])
            iqa_method, transform_mode = methodloader()

            # load dataset: img + y
            dataloader = DataLoader(dataset, folder_path[dataset], img_num[dataset], transform_mode=transform_mode)
            data = dataloader.get_data()

            # create framework
            frame = feature_framework(iqa_method, backbone=configs['backbone'])

            # extract multiscale features
            X, y = frame.multiscale_loader(data)
 
            # save matrix
            matrix = torch.cat((X, y), dim=1)
            save_matrix(matrix, configs['multiscale_dir'] + dataset + '.pt')
            # save matrix to txt, for debugging, delete it after debugging
            np.savetxt(configs['multiscale_dir'] + dataset + '.txt', np.array(matrix), fmt='%f', delimiter='\t')
        else:
            matrix = load_matrix(configs['multiscale_dir'] + dataset + '.pt')
            X, y = matrix[:,:-1], matrix[:,-1]

        y = normalize_y(y, configs['curd']['datasets'][dataset_id])
        X = normalize_X(X, configs['curd']['norm_Rs'][dataset_id])

        X_list.append(expand(X))
        y_list.append(y.unsqueeze(1))

    curd = CURD(torch.cat(X_list, dim=0), torch.cat(y_list, dim=0).squeeze(1), no=configs['curd']['curd_no'], output_file_name=configs['curd_dir']+'curd_temp.txt')

    # remove curd temp file
    if configs['curd']['rm_temp'] and os.path.exists(curd.get_output_file_name()):
        print('remove curd temp files...')
        os.remove(curd.get_output_file_name())
    
    # load curd temp file
    if os.path.exists(curd.get_output_file_name()):
        curd_outputs = np.loadtxt(curd.get_output_file_name())
    else:
        curd_outputs = curd.process(configs['curd']['save_num']) # TODO: 内部函数使用np，适当重构为torch
    curd_outputs = torch.tensor(curd_outputs)

    # perform regression evaluation and save data
    logs = torch.zeros((configs['curd']['save_num'], 11)) # sw(0) srcc(1-4) plcc(5-8) sum/8(9) epoch(10)
    parameter_paths = []
    n = len(configs['curd']['datasets'])
    for epoch, row in tqdm(enumerate(curd_outputs), total=len(curd_outputs)):
        plccs, srccs, beta_matrix = torch.zeros(n), torch.zeros(n), torch.zeros((n, 7))
        for i, X in enumerate(X_list):
            index = row[:configs['curd']['curd_no']].to(torch.int)
            beta_matrix[i] = regression(X, y_list[i], index)
            y_hat = prediction(X, beta_matrix[i], index)
            plccs[i], srccs[i], err_flag = calculate_sp(y_list[i].squeeze(), y_hat.squeeze(), show=False)
            if err_flag:
                break

        # log and save parameter
        if not err_flag:
            logs[epoch] = torch.cat((row[configs['curd']['curd_no']].unsqueeze(0), plccs, srccs, torch.tensor([(plccs.sum() + srccs.sum()) / 8]), torch.tensor([epoch])))
        
        parameter_path = configs['ckpt_dir'] + configs['curd']['curd_file'][:-3] + '_' + str(epoch) + '.pt'
        parameter_paths.append(parameter_path)
        save_parameter(index, beta = beta_matrix, file_path = parameter_path, show = False)

    # save logs
    save_logs(logs, configs['curd_dir'] + configs['curd']['log_file'], configs['curd']['save_num'], sort_num = 9)
    print('The curd training is finished!')


def enhanced_process(configs):
    if configs['enhanced']['multiscale_flag']:
        # load iqa method
        methodloader = MethodLoader(configs['method'], configs['pyiqa'], configs['enhanced']['ckpt'])
        iqa_method, transform_mode = methodloader()

        # load dataset: img + y
        dataLoader = DataLoader(configs['enhanced']['dataset'], folder_path[configs['enhanced']['dataset']], img_num[configs['enhanced']['dataset']], transform_mode = transform_mode)
        data = dataLoader.get_data()

        # create framework
        frame = feature_framework(iqa_method, backbone=configs['backbone'])

        # evaluate the model
        X, y = frame.multiscale_loader(data)
    else:
        matrix = load_matrix(configs['multiscale_dir'] + configs['enhanced']['dataset'] + '.pt')
        X, y = matrix[:, :-1], matrix[:, -1]

    # load parameters: index, beta
    index, beta = load_parameter(configs['ckpt_dir'] + configs['enhanced']['curd_file'])
    if beta.shape[0] != 1:
        index, beta = index, beta[0]
    beta_index_to_function(index, beta)

    # prediction
    X = (expand(normalize_X(X, configs['enhanced']['norm_R'])) if configs['enhanced']['norm_R'] is not None else expand(X))
    y = normalize_y(y, configs['enhanced']['dataset'])
    y_hat = prediction(X, beta, index)
    calculate_sp(y, y_hat.squeeze())

    # save matrix
    np.savetxt(configs['enhanced_dir'] + configs['enhanced']['enhanced_file'], np.column_stack((y.squeeze(), y_hat.squeeze())), fmt='%f', delimiter='\t')


if __name__ == '__main__':
    '''
        mode: `
            origin    : original version of the method.
            curd      : curd training framework.
            enhanced  : enhanced version of the method.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.filterwarnings('ignore')

    # load yaml file as configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        configs = OmegaConf.load(file)

    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    create_directory(configs['multiscale_dir'])
    create_directory(configs['ckpt_dir'])
    create_directory(configs['curd_dir'])
    create_directory(configs['enhanced_dir'])

    if configs['mode'] == 'origin':
        origin_process(configs)
    elif configs['mode'] == 'curd':
        curd_process(configs)
    elif configs['mode'] == 'enhanced':
        enhanced_process(configs)
    else:
        print('The mode is not supported.')
        exit(0)
    
    