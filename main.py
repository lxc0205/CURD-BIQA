import os
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from method_loader import MethodLoader
from framework import feature_framework
from data.dataLoader import DataLoader, normalize_X, normalize_y, folder_path, img_num
from curd import CURD, calculate_sp, regression, prediction, expand, beta_index_to_function
from utils import save_parameter, load_parameter, save_matrix, load_matrix, save_logs, create_directory


def train(configs):
    # load multiscale score
    X_list, y_list = [], []

    for dataset_id, dataset in enumerate(configs['train']['datasets']):
        if configs['train']['multiscale_flag']:
            # load iqa method
            iqa_method = MethodLoader(configs['method'], configs['pyiqa'], configs['train']['ckpts'][dataset_id])()

            # load dataset: img + y
            data = DataLoader(dataset, folder_path[dataset], img_num[dataset])()

            # create framework
            frame = feature_framework(iqa_method, backbone=configs['backbone'])

            # extract multiscale features
            X, y = frame.multiscale_loader(data)

            y = normalize_y(y, configs['train']['datasets'][dataset_id])
            X = normalize_X(X)
 
            # save matrix
            matrix = torch.cat((X, y), dim=1)
            save_matrix(matrix, configs['multiscale_dir'] + dataset + '.pt')
            # save matrix to txt, for debugging, delete it after debugging
            np.savetxt(configs['multiscale_dir'] + dataset + '.txt', np.array(matrix), fmt='%f', delimiter='\t')
        else:
            matrix = load_matrix(configs['multiscale_dir'] + dataset + '.pt')
            X, y = matrix[:,:-1], matrix[:,-1]

        X_list.append(expand(X))
        y_list.append(y)

    curd = CURD(torch.cat(X_list, dim=0), torch.cat(y_list, dim=0), no=configs['train']['curd_no'], output_file_name=configs['curd_dir']+'curd_temp.txt')

    # remove curd temp file
    if configs['train']['rm_temp'] and os.path.exists(curd.get_output_file_name()):
        print('remove curd temp files...')
        os.remove(curd.get_output_file_name())
    
    # load curd temp file
    if os.path.exists(curd.get_output_file_name()):
        curd_outputs = np.loadtxt(curd.get_output_file_name())
    else:
        curd_outputs = curd.process(configs['train']['save_num'])
    curd_outputs = torch.tensor(curd_outputs)

    # perform regression evaluation and save data
    logs = torch.zeros((configs['train']['save_num'], 11)) # sw(0) srcc(1-4) plcc(5-8) sum/8(9) epoch(10)
    parameter_paths = []
    n = len(configs['train']['datasets'])
    for epoch, row in tqdm(enumerate(curd_outputs), total=len(curd_outputs)):
        plccs, srccs, beta_matrix = torch.zeros(n), torch.zeros(n), torch.zeros((n, configs['train']['curd_no']))
        for i, X in enumerate(X_list):
            index = row[:configs['train']['curd_no']].to(torch.int)
            beta_matrix[i] = regression(X, y_list[i], index)
            y_hat = prediction(X, beta_matrix[i], index)
            plccs[i], srccs[i], err_flag = calculate_sp(y_list[i].squeeze(), y_hat.squeeze(), show=False)
            if err_flag:
                break

        # log and save parameter
        if not err_flag:
            logs[epoch] = torch.cat((row[configs['train']['curd_no']].unsqueeze(0), plccs, srccs, torch.tensor([(plccs.sum() + srccs.sum()) / 8]), torch.tensor([epoch])))
        
        parameter_path = configs['ckpt_dir'] + configs['train']['curd_file'][:-3] + '_' + str(epoch) + '.pt'
        parameter_paths.append(parameter_path)
        save_parameter(index, beta = beta_matrix, file_path = parameter_path, show = False)

    # save logs
    save_logs(logs, configs['curd_dir'] + configs['train']['log_file'], configs['train']['save_num'], sort_num = 9)
    print('The curd training is finished!')


def evaluate(configs):
    # print method name and dataset name
    print(f"Evaluating method: {configs['method']},\tEvaluating dataset: {configs['evaluate']['dataset']}")

    # load iqa method
    iqa_method = MethodLoader(configs['method'], configs['pyiqa'], configs['evaluate']['ckpt'])()

    # load dataset: img + y
    data = DataLoader(configs['evaluate']['dataset'], folder_path[configs['evaluate']['dataset']], img_num[configs['evaluate']['dataset']])()

    # create framework
    frame = feature_framework(iqa_method)  

    # evaluate the model
    X, y = frame.origin_loader(data)
    print('Origin method srcc and plcc:')
    calculate_sp(X, y)

    if configs['evaluate']['multiscale_flag']:
        # create framework
        frame = feature_framework(iqa_method, backbone=configs['backbone'])

        # evaluate the model
        X, y = frame.multiscale_loader(data)

        y = normalize_y(y, configs['evaluate']['dataset'])
        X = normalize_X(X)
        
    else:
        matrix = load_matrix(configs['multiscale_dir'] + configs['evaluate']['dataset'] + '.pt')
        X, y = matrix[:, :-1], matrix[:, -1]

    # load parameters: index, beta
    index, beta = load_parameter(configs['ckpt_dir'] + configs['evaluate']['curd_file'])
    if beta.shape[0] != 1:
        index, beta = index, beta[0]

    # prediction
    y_hat = prediction(expand(X), beta, index)
    print('Enhanced method srcc and plcc:')
    calculate_sp(y.squeeze(), y_hat.squeeze())

    # show the model and its latex expression
    # beta_index_to_function(index, beta)

    # save matrix
    np.savetxt(configs['enhanced_dir'] + configs['evaluate']['enhanced_file'], np.column_stack((y.squeeze(), y_hat.squeeze())), fmt='%f', delimiter='\t')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.filterwarnings('ignore')

    # load yaml file as configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        configs = OmegaConf.load(file)

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    create_directory(configs['multiscale_dir'])
    create_directory(configs['ckpt_dir'])
    create_directory(configs['curd_dir'])
    create_directory(configs['enhanced_dir'])

    if configs['train_mode']:
        train(configs)
    else:
        evaluate(configs)
       
    