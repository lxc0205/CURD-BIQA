import torch
import numpy as np
from math import comb
from tqdm import tqdm
from scipy.linalg import det
from itertools import combinations


class CURD:
    def __init__(self, Mssim, mos, no = 7, output_file_name='./outputs/curd_temp.txt'):
        # 数据
        self.Mssim_expand, self.mos = Mssim, mos
        # 输入输出文件名
        self.output_file_name = output_file_name
        # 遍历范围
        self.R_min, self.R_max = 0, self.Mssim_expand.shape[1]-7
        # 其他常数
        self.NO = no
        self.THRES = 0.9999

    def pearson_corr(self):
         self.correlation_matrix = np.corrcoef(np.concatenate((self.Mssim_expand, self.mos), axis=1).T)

    def process(self, save_num):
        self.pearson_corr()
        Rhere = np.zeros((self.NO+1, self.NO+1))
        Rxy = np.zeros((self.NO+1, self.NO+1))
        n = self.Mssim_expand.shape[1]
        temp = 0
        for i in range(self.NO+1):
            Rhere[i][i] = 1

        with open(self.output_file_name, 'w') as file:
            for i1 in tqdm(range(self.R_min, self.R_max+1)):
                Rhere[0][self.NO] = self.correlation_matrix[i1][n]
                for i2 in range(i1+1, n):
                    temp = self.correlation_matrix[i1][i2]
                    if temp > self.THRES:
                        continue
                    Rhere[0][1]=temp
                    Rhere[1][self.NO] = self.correlation_matrix[i2][n]
                    for i3 in range(i2+1, n):
                        temp = self.correlation_matrix[i1][i3]
                        if temp > self.THRES:
                            continue
                        Rhere[0][2] = temp
                        temp = self.correlation_matrix[i2][i3]
                        if temp > self.THRES:
                            continue
                        Rhere[1][2] = temp
                        Rhere[2][self.NO] = self.correlation_matrix[i3][n]
                        for i4 in range(i3+1, n):
                            temp = self.correlation_matrix[i1][i4]
                            if temp > self.THRES:
                                continue
                            Rhere[0][3] = temp
                            temp = self.correlation_matrix[i2][i4]
                            if temp > self.THRES:
                                continue
                            Rhere[1][3] = temp
                            temp = self.correlation_matrix[i3][i4]
                            if temp > self.THRES:
                                continue
                            Rhere[2][3] = temp
                            Rhere[3][self.NO] = self.correlation_matrix[i4][n]
                            for i5 in range(i4+1, n):
                                temp = self.correlation_matrix[i1][i5]
                                if temp > self.THRES:
                                    continue
                                Rhere[0][4] = temp
                                temp = self.correlation_matrix[i2][i5]
                                if temp > self.THRES:
                                    continue
                                Rhere[1][4] = temp
                                temp = self.correlation_matrix[i3][i5]
                                if temp > self.THRES:
                                    continue
                                Rhere[2][4] = temp
                                temp = self.correlation_matrix[i4][i5]
                                if temp > self.THRES:
                                    continue
                                Rhere[3][4] = temp
                                Rhere[4][self.NO] = self.correlation_matrix[i5][n]
                                for i6 in range(i5+1, n):
                                    temp = self.correlation_matrix[i1][i6]
                                    if temp > self.THRES:
                                        continue
                                    Rhere[0][5] = temp
                                    temp = self.correlation_matrix[i2][i6]
                                    if temp > self.THRES:
                                        continue
                                    Rhere[1][5] = temp
                                    temp = self.correlation_matrix[i3][i6]
                                    if temp > self.THRES:
                                        continue
                                    Rhere[2][5] = temp
                                    temp = self.correlation_matrix[i4][i6]
                                    if temp > self.THRES:
                                        continue
                                    Rhere[3][5] = temp
                                    temp = self.correlation_matrix[i5][i6]
                                    if temp > self.THRES:
                                        continue
                                    Rhere[4][5] = temp
                                    Rhere[5][self.NO] = self.correlation_matrix[i6][n]
                                    for i7 in range(i6+1, n):
                                        temp = self.correlation_matrix[i1][i7]
                                        if temp > self.THRES:
                                            continue
                                        Rhere[0][6] = temp
                                        temp = self.correlation_matrix[i2][i7]
                                        if temp > self.THRES:
                                            continue
                                        Rhere[1][6] = temp
                                        temp = self.correlation_matrix[i3][i7]
                                        if temp > self.THRES:
                                            continue
                                        Rhere[2][6] = temp
                                        temp = self.correlation_matrix[i4][i7]
                                        if temp > self.THRES:
                                            continue
                                        Rhere[3][6] = temp
                                        temp = self.correlation_matrix[i5][i7]
                                        if temp > self.THRES:
                                            continue
                                        Rhere[4][6] = temp
                                        temp = self.correlation_matrix[i6][i7]
                                        if temp > self.THRES:
                                            continue
                                        Rhere[5][6] = temp
                                        Rhere[6][self.NO] = self.correlation_matrix[i7][n]
                                        
                                        for i in range(0, self.NO+1):
                                            for j in range(0, self.NO+1):
                                                Rxy[i][j] = Rhere[i][j]

                                        for i in range(self.NO):
                                            recidiag = 1 / Rxy[i][i]
                                            for j in range(i+1, self.NO+1):
                                                temp = Rxy[i][j] * recidiag
                                                for p in range(j, self.NO+1):
                                                    m = Rxy[j][p] - Rxy[i][p] * temp
                                                    Rxy[j][p] = Rxy[j][p] - Rxy[i][p] * temp
                                        
                                        sw = Rxy[self.NO][self.NO] # 无符号非相关系数的平方 omega^2
                                        
                                        if 0 < sw <= 1:
                                            file.write(str(i1) + ' ' + str(i2) + ' ' + str(i3) + ' ' + str(i4) + ' ' + str(i5) + ' ' + str(i6) + ' ' + str(i7) + ' ' + str(sw) + '\n')
        
        #从文件里读入
        mat = []
        with open(self.output_file_name, 'r') as file:
            for line in file:
                if line.strip():
                    temp = [float(x) for x in line.split()]
                    mat.append(temp)
        mat = np.array(mat)

        # Sort matrix and save results
        sorted_matrix = sort(mat, order="ascending", row = 7)
        sorted_matrix = sorted_matrix[:save_num, :]

        with open(self.output_file_name, 'w') as file:
            for i in range(sorted_matrix.shape[0]):
                mat = sorted_matrix[i,:]
                for j in range(len(mat)):
                    file.write(f"{int(mat[j]) if j < self.NO else mat[j]}" + '\t')
                file.write('\n')

        return sorted_matrix

    def process_det(self, save_num):
        self.pearson_corr()
        variable_num = self.correlation_matrix.shape[0] - 1
        comb_num = comb(variable_num, self.NO)
        matrix = np.zeros((comb_num, self.NO + 1))
        # Calculate submatrices and their determinants
        epoch = 0
        for combo in tqdm(combinations(range(variable_num), self.NO), total = comb_num):
            # Calculate determinants
            submatrix_den = self.correlation_matrix[combo, :][:, combo] # 计算square_omega的分母
            if (submatrix_den > 0.9999).sum() > self.NO: 
                continue
            submatrix_num = self.correlation_matrix[combo + (variable_num, ), :][:, combo + (variable_num, )] # 计算square_omega的分子

            # Store results
            matrix[epoch] = np.concatenate((combo, [det(submatrix_num) / det(submatrix_den) if det(submatrix_den) != 0 else 1])) # store index + square_omega
            epoch += 1
        print(f'Number of curd items: {epoch}\n')

        # Sort matrix and save results
        sorted_matrix = sort(matrix[:epoch, :], order="ascending", row = 7)
        sorted_matrix = sorted_matrix[:save_num, :]

        return sorted_matrix
    
    def get_output_file_name(self):
        return self.output_file_name

def sort(data, order, row):
    sorted_indices = np.argsort(data[:, row], axis=0, kind='mergesort')
    if order == 'descending':
        sorted_indices = sorted_indices[::-1]
    sorted_indices = np.tile(sorted_indices.reshape(-1, 1), (1, data.shape[1]))
    sorted_matrix = np.take_along_axis(data, sorted_indices, axis=0)
    return sorted_matrix

def expand(Mssim):
    if torch.isnan(Mssim).any(): print("origin function warning: Tensor contains NaN.")
    if torch.isinf(Mssim).any(): print("origin function warning: Tensor contains Inf.")

    Mssim_expand = torch.cat((
        Mssim, Mssim**2, torch.sqrt(Mssim), Mssim**3, Mssim**(1/3), torch.log(Mssim+1)/torch.log(torch.tensor(2.0)), torch.pow(2, Mssim)-1, (torch.exp(Mssim)-1) /(torch.exp(torch.tensor(1.0))-1)), dim=1)
    if torch.isnan(Mssim_expand).any(): print("expand function warning: Tensor contains NaN.")
    if torch.isinf(Mssim_expand).any(): print("expand function warning: Tensor contains Inf.")

    return Mssim_expand



