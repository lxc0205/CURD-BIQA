import os
import argparse
import numpy as np
from tqdm import tqdm
from data_loader import loadMssimMos
from curd import CURD, expand, calculate_sp, regression, prediction, sort

# train mode 1: train and test on all datasets
def regression1(inputFileSet, outputFile, NO, sorted_matrix, SAVE_NUM):
    # edit：for certain methods，evalued by framework.py
    Mssim_list = []
    mos_list = []
    for dataset in inputFileSet:
        Mssim_temp, mos_temp = loadMssimMos({dataset})
        Mssim_list.append(expand(Mssim_temp, 'mode2'))
        mos_list.append(mos_temp)
    matrix = np.zeros((SAVE_NUM, 2*NO + 10))
    for epoch, row in tqdm(enumerate(sorted_matrix), total=len(sorted_matrix)):
        plcc_list = [0]*4
        srcc_list = [0]*4
        for i in range(len(Mssim_list)):
            index = row[:NO].astype(int)
            beta = regression(Mssim_list[i], mos_list[i], index)
            yhat = prediction(Mssim_list[i], beta, index)
            plcc_list[i], srcc_list[i] = calculate_sp(mos_list[i].squeeze(), yhat.squeeze()) 
        # 0 1 2 3 4 5 6    7    8 9 10 11 12 13 14    15 16 17 18     19 20 21 22             23
        # ----index----   sw    -------beta-------        srcc            plcc              sum/8
        matrix[epoch] = np.concatenate((row[:NO+1], beta.squeeze(), plcc_list, srcc_list,[(sum(plcc_list)+sum(srcc_list))/8]))
    print(f'Number of regression items: {epoch}\n')
    # 排序,结果保存到文件
    matrix = sort(matrix, order="descending", row = 23)[:SAVE_NUM, :]
    np.savetxt(outputFile, matrix, fmt=['%d']*NO + ['%f']*(matrix.shape[1]-NO), delimiter='\t')

# train mode 2: 
def regression2(inputFileSet, outputFile, NO, sorted_matrix, SAVE_NUM):
    # edit：for certain methods，evalued by framework.py
    plcc_baseline = [0.7590450564940359, 0.8443112861149688, 0.7009730564389459, 0.9071180076951416]
    srcc_baseline = [0.7194202223850794, 0.8603025984232107, 0.6317942226022212, 0.8955912156972166]
    Mssim_list = []
    mos_list = []
    for dataset in inputFileSet:
        Mssim_temp, mos_temp = loadMssimMos({dataset})
        Mssim_list.append(expand(Mssim_temp, 'mode2'))
        mos_list.append(mos_temp)
    matrix = np.zeros((SAVE_NUM, 2*NO + 31))
    for epoch, row in tqdm(enumerate(sorted_matrix), total=len(sorted_matrix)):
        plcc_list = [0]*4
        srcc_list = [0]*4
        beta_mat = [[0]*7]*4
        for i in range(len(Mssim_list)):
            index = row[:NO].astype(int)
            beta_mat[i] = regression(Mssim_list[i], mos_list[i], index)
            yhat = prediction(Mssim_list[i], beta_mat[i], index)
            plcc_list[i], srcc_list[i] = calculate_sp(mos_list[i].squeeze(), yhat.squeeze())
        plcc_diff = [plcc - plcc_baseline[i] for i, plcc in enumerate(plcc_list)]
        srcc_diff = [srcc - srcc_baseline[i] for i, srcc in enumerate(srcc_list)] 
        if all(x > 0 for x in plcc_diff) and all(x > 0 for x in srcc_diff):
            # 0 1 2 3 4 5 6    7    8 - 14    15 - 21    22 - 28    29 - 35    36 37 38 39        40 41 42 43            44
            # ----index----   sw     beta1     beta2      beta3      beta4         srcc               plcc              sum/8
            matrix[epoch] = np.concatenate((row[:NO+1], beta_mat[0].squeeze(), beta_mat[1].squeeze(), beta_mat[2].squeeze(), beta_mat[3].squeeze(), plcc_list, srcc_list,[(sum(plcc_list)+sum(srcc_list))/8]))
        else:
            matrix[epoch] = np.zeros(matrix.shape[1])
    print(f'Number of regression items: {epoch}\n')
    # 排序,结果保存到文件
    matrix = sort(matrix, order="descending", row = 44)[:SAVE_NUM, :]
    np.savetxt(outputFile, matrix, fmt=['%d']*NO + ['%f']*(matrix.shape[1]-NO), delimiter=' ')

def main(config):
    # Load data
    inputFileSet = [f'./Outputs/{config.method}/multiscale outputs/' + item for item in config.inputFileSet]
    outputFile = f'./Outputs/{config.method}/curd outputs/' + config.outputFile
    if not os.path.exists(f'./Outputs/{config.method}/curd outputs/'):
        os.makedirs(f'./Outputs/{config.method}/curd outputs/')

    Mssim ,mos = loadMssimMos(inputFileSet)
    temp_file_path = './outputs/curd_temp.txt'
    curd = CURD(Mssim, mos.squeeze(1), temp_file_path)
    sorted_matrix = curd.process(config.save_num)
    
    # Perform regression evaluation and save data
    # regression1(inputFileSet, outputFile, temp_file_path, curd.NO, sorted_matrix, config.save_num)
    regression2(inputFileSet, outputFile, temp_file_path, curd.NO, sorted_matrix, config.save_num)

    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

if __name__ == "__main__":
    print(f'The curd iqa process has started...\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFileSet',dest='inputFileSet', type=str, required=True, nargs='+', help='Input file set.')
    parser.add_argument('--outputFile', dest='outputFile', type=str, required=True, default='./curd_fitting.txt', help='Output flie dir.')
    parser.add_argument('--method', dest='method', type=str, required=True, default='dbcnn', help='Support methods: clipiqa|clipiqa+|ilniqe|wadiqam_nr|dbcnn|paq2piq|hyperiqa|tres|tres-flive|tres-koniq|maniqa|maniqa-kadid|maniqa-koniq|maniqa-pipal')
    parser.add_argument('--save_num', dest='save_num', type=int, default=500000, help='Save numbers.')
    config = parser.parse_args()
    print(f'Basic method:{config.method},\ttraining Files:{config.inputFileSet},\tOutput File:{config.outputFile}')
    main(config)

    print(f'The curd iqa finished!')
