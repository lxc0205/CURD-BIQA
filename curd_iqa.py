import os
import argparse
import numpy as np
from tqdm import tqdm
from data_loader import loadMssimMos
from curd import CURD, expand2, calculate_sp, regression, prediction, sort

def main(config):
    SAVE_NUM = config.save_num
    # Load data
    inputFileSet = [f'./Outputs/{config.method}/multiscale outputs/' + item for item in config.inputFileSet]
    outputFile = f'./Outputs/{config.method}/curd outputs/' + config.outputFile
    if not os.path.exists(f'./Outputs/{config.method}/curd outputs/'):
        os.makedirs(f'./Outputs/{config.method}/curd outputs/')

    Mssim ,mos = loadMssimMos(inputFileSet)
    temp_file_path = './outputs/curd_temp.txt'
    curd = CURD(Mssim, mos.squeeze(1), temp_file_path)
    sorted_matrix = curd.process(SAVE_NUM)
    
    # Perform regression evaluation and save data
    Mssim_list = []
    mos_list = []
    for dataset in inputFileSet:
        Mssim_temp, mos_temp = loadMssimMos({dataset})
        Mssim_list.append(expand2(Mssim_temp))
        mos_list.append(mos_temp)

    matrix = np.zeros((SAVE_NUM, 2*curd.NO + 10))
    for epoch, row in tqdm(enumerate(sorted_matrix), total=len(sorted_matrix)):
        plcc_list = [0, 0, 0, 0]
        srcc_list = [0, 0, 0, 0]
        for i in range(len(Mssim_list)):
            index = row[:curd.NO].astype(int)
            beta = regression(Mssim_list[i], mos_list[i], index)
            yhat = prediction(Mssim_list[i], beta, index)
            plcc_list[i], srcc_list[i] = calculate_sp(mos_list[i].squeeze(), yhat.squeeze())
        # 0 1 2 3 4 5 6    7    8 9 10 11 12 13 14    15 16 17 18     19 20 21 22             23
        # ----index----   sw    -------beta-------        srcc            plcc              sum/8
        matrix[epoch] = np.concatenate((row[:curd.NO+1], beta.squeeze(), plcc_list, srcc_list,[(sum(plcc_list)+sum(srcc_list))/8]))
    print(f'Number of regression items: {epoch}\n')

    # 排序,结果保存到文件
    matrix = sort(matrix, order="descending", row = 23)[:SAVE_NUM, :]
    np.savetxt(outputFile, matrix, fmt=['%d']*curd.NO + ['%f']*(matrix.shape[1]-curd.NO), delimiter='\t')

    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

if __name__ == "__main__":
    print(f'The curd iqa process has started...\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFileSet',dest='inputFileSet', type=str, nargs='+', help='Input file set.')
    parser.add_argument('--outputFile', dest='outputFile', type=str, default='./curd_fitting.txt', help='Output flie dir.')
    parser.add_argument('--method', dest='method', type=str, default='dbcnn', help='Support methods: clipiqa|clipiqa+|ilniqe|wadiqam_nr|dbcnn|paq2piq|hyperiqa|tres|tres-flive|tres-koniq|maniqa|maniqa-kadid|maniqa-koniq|maniqa-pipal')
    parser.add_argument('--save_num', dest='save_num', type=int, default=500000, help='Save numbers.')
    config = parser.parse_args()
    print(f'Basic method:{config.method},\ttraining Files:{config.inputFileSet},\tOutput File:{config.outputFile}')
    main(config)

    print(f'The curd iqa finished!')
