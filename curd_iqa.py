import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from data_loader import loadMssimMos
from curd import CURD, expand, calculate_sp, regression, prediction, sort


# BL = np.array([0.9680,0.9610,0.9830,0.9820,0.9430,0.9370,0.9460,0.9300])
BL_plcc = np.array([0.9680,0.9830,0.9430,0])
BL_srcc = np.array([0.9610,0.9820,0.9370,0])

def regressing(inputFileSet, outputFile, NO, sorted_matrix, savenum, norm_R_set):
    Mssim_list, mos_list = [], []
    for id, dataset in enumerate(inputFileSet):
        Mssim_temp, mos_temp = loadMssimMos({dataset}, [norm_R_set[id]])
        Mssim_list.append(expand(Mssim_temp))
        mos_list.append(mos_temp)
    # matrix 结构:
    # 0 1 2 3 4 5 6    7    8 - 14    15 - 21    22 - 28    29 - 35    36 37 38 39    40 41 42 43     44
    # ----index----   sw    beta1      beta2      beta3      beta4         srcc           plcc       sum/8
    matrix = np.zeros((savenum, 2*NO + 31))
    for epoch, row in tqdm(enumerate(sorted_matrix), total=len(sorted_matrix)):
        plcc_list, srcc_list, beta_mat = [0]*4, [0]*4, [[0]*7]*4
        for i in range(len(Mssim_list)):
            index = row[:NO].astype(int)
            beta_mat[i] = regression(Mssim_list[i], mos_list[i], index)
            yhat = prediction(Mssim_list[i], beta_mat[i], index)
            plcc_list[i], srcc_list[i] = calculate_sp(mos_list[i].squeeze(), yhat.squeeze())

        rounded_plcc_list = np.round(plcc_list, decimals=3)
        rounded_srcc_list = np.round(srcc_list, decimals=3)
        plcc_diff = [plcc - BL_plcc[i] for i, plcc in enumerate(rounded_plcc_list)]
        srcc_diff = [srcc - BL_srcc[i] for i, srcc in enumerate(rounded_srcc_list)]
        if all(x >= 0 for x in plcc_diff) and all(x >= 0 for x in srcc_diff):
            matrix[epoch] = np.concatenate((row[:NO+1], beta_mat[0].squeeze(), beta_mat[1].squeeze(), beta_mat[2].squeeze(), beta_mat[3].squeeze(), plcc_list, srcc_list,[(sum(plcc_list)+sum(srcc_list))/8]))
    print(f'Number of regression items: {epoch}\n')
    # sort and save into a file
    matrix = sort(matrix, order="descending", row = 44)[:savenum, :]
    np.savetxt(outputFile, matrix, fmt=['%d']*NO + ['%f']*(matrix.shape[1]-NO), delimiter=' ')

def main(config):
    # Load data
    output_Path = f'./outputs/{config.method}/'
    inputFileSet = [output_Path + 'multiscale outputs/' + item for item in config.inputFile_set]
    outputFile = output_Path + 'curd outputs/' + config.outputFile
    if not os.path.exists(output_Path + 'curd outputs/'):
        os.makedirs(output_Path + 'curd outputs/')

    Mssim ,mos = loadMssimMos(inputFileSet)

    temp_file = output_Path + 'curd_temp.txt'
    curd = CURD(Mssim, mos.squeeze(1), temp_file)
    if os.path.exists(temp_file):
        sorted_matrix = np.loadtxt(temp_file)
    else:
        sorted_matrix = curd.process(config.save_num)
        
    # Perform regression evaluation and save data
    regressing(inputFileSet, outputFile, curd.NO, sorted_matrix, config.save_num, config.norm_R_set)

    if config.rm_cache:
        print(f'Remove cache files...')
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', dest='method', type=str, required=True, default='dbcnn', help='Support methods: clipiqa+|wadiqam_nr|dbcnn|paq2piq|hyperiqa|tres|tres-flive|tres-koniq|maniqa|maniqa-kadid|maniqa-koniq|maniqa-pipal')
    parser.add_argument('--norm_R_set', dest='norm_R_set', type=int, nargs='+', default=None, help='The range of mssim normalization.')
    parser.add_argument('--inputFile_set',dest='inputFile_set', type=str, required=True, nargs='+', help='Input file set.')
    parser.add_argument('--outputFile', dest='outputFile', type=str, required=True, default='./curd_fitting.txt', help='Output flie path.')
    parser.add_argument('--save_num', dest='save_num', type=int, default=500000, help='Save numbers.')
    parser.add_argument('--rm_cache', dest='rm_cache', action='store_true', help='The flag of remove temp files curd_temp.txt.')
    config = parser.parse_args()
    print(f'The curd iqa process has started...\n')

    # print configs
    config_dict = vars(config)
    print("Configs:")
    for key, value in sorted(config_dict.items()):
        print(f"{key.replace('_', ' ').title()}: {value}")

    # print time
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    
    main(config)

    print(f'The curd iqa finished!')
