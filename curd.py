import numpy as np
from math import comb
from tqdm import tqdm
from scipy.linalg import det
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr, pearsonr


class CURD:
    def __init__(self, Mssim, mos, output_flie_name='./outputs/curd_temp.txt'):
        # 数据
        self.Mssim, self.mos = Mssim, mos
        # 输入输出文件名
        self.output_flie_name = output_flie_name
        # 遍历范围
        self.R_min, self.R_max = 0, 8*self.Mssim.shape[1]-7
        # 其他常数
        self.NO = 7
        self.THRES = 0.9999

    def pearson_corr(self):
         self.correlation_matrix = np.corrcoef(np.concatenate((self.Mssim_expand, self.mos[:, np.newaxis]), axis=1).T)

    def process(self, save_num):
        self.Mssim_expand = expand(self.Mssim)
        self.pearson_corr()
        Rhere = np.zeros((self.NO+1, self.NO+1))
        Rxy = np.zeros((self.NO+1, self.NO+1))
        n = self.Mssim_expand.shape[1]
        temp = 0
        for i in range(self.NO+1):
            Rhere[i][i] = 1

        with open(self.output_flie_name, 'w') as file:
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
        with open(self.output_flie_name, 'r') as file:
            for line in file:
                if line.strip():
                    temp = [float(x) for x in line.split()]
                    mat.append(temp)
        mat = np.array(mat)

        # Sort matrix and save results
        sorted_matrix = sort(mat, order="ascending", row = 7)
        sorted_matrix = sorted_matrix[:save_num, :]

        with open(self.output_flie_name, 'w') as file:
            for i in range(sorted_matrix.shape[0]):
                mat = sorted_matrix[i,:]
                for j in range(len(mat)):
                    file.write(f"{int(mat[j]) if j < self.NO else mat[j]}" + '\t')
                file.write('\n')

        return sorted_matrix

    def process_det(self, save_num):
        self.Mssim_expand = expand(self.Mssim)
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

def sort(data, order, row):
    sorted_indices = np.argsort(data[:, row], axis=0, kind='mergesort')
    if order == 'descending':
        sorted_indices = sorted_indices[::-1]
    sorted_indices = np.tile(sorted_indices.reshape(-1, 1), (1, data.shape[1]))
    sorted_matrix = np.take_along_axis(data, sorted_indices, axis=0)
    return sorted_matrix

def regression(Mssim, mos, index):
    Mssim_s = Mssim[:, index]
    U, S, Vt = np.linalg.svd(Mssim_s, full_matrices=False)
    inv_Mssim_s = Vt.T @ np.diag(1 / S) @ U.T
    beta = inv_Mssim_s @ mos
    return beta

def prediction(Mssim, beta, index):
    Mssim_s = Mssim[:, index]
    yhat = Mssim_s @ beta
    return yhat

def plot_y_yhat(pred, label):
    fig, ax = plt.subplots()
    ax.plot(range(0, pred.shape[0]), label, label='True Values', marker='o')
    ax.plot(range(0, pred.shape[0]), pred, label='Predicted Values', marker='x')
    ax.set_title('Comparison of True and Predicted Values')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()
    print(f'mse: {np.mean((pred - label) ** 2)}')
    plt.show()

def calculate_sp(y, yhat):
    plcc, p_PLCC = pearsonr(y, yhat)
    srcc, p_SRCC = spearmanr(y, yhat)
    if p_PLCC >0.05:
        print("The plcc correlation is not significant.")
    if p_SRCC >0.05:
        print("The srcc correlation is not significant.")
    return np.abs(plcc), np.abs(srcc)

def expand(Mssim):
    Mssim_expand = np.hstack((
        Mssim,  Mssim**2,   np.sqrt(Mssim), Mssim**3,   Mssim**(1/3),   np.log(Mssim+1) / np.log(2),    np.power(2, Mssim) - 1, (np.exp(Mssim)-1) / (np.exp(1)-1)
    #     Mssim,  Mssim**2,   np.sqrt(Mssim), Mssim**3,   Mssim**(1/3),   np.log(Mssim),                  np.power(2, Mssim),     np.exp(Mssim)
    ))

    if np.isnan(Mssim_expand).any(): print("expand function warning: Array contains NaN.")
    if np.isinf(Mssim_expand).any(): print("expand function warning: Array contains Inf.")

    return Mssim_expand

def index2func(index, data_dim = 6):
    value_list = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'] # decide by col
    value_latex_list = ['{\\textbf{x}_0}', '{\\textbf{x}_1}', '{\\textbf{x}_2}', '{\\textbf{x}_3}', '{\\textbf{x}_4}', '{\\textbf{x}_5}'] # decide by col
    row = index // data_dim
    col = index % data_dim

    value = value_list[col]
    value_latex = value_latex_list[col]
    if row == 0:
        func = value
        func_latex = value_latex
    elif row == 1:
        func = value + '^2'
        func_latex = value_latex + '^2'
    elif row == 2:
        func = 'sqrt(' + value + ')'
        func_latex = '\\sqrt{' + value_latex + '}'
    elif row == 3:
        func = value + '^3'
        func_latex = value_latex + '^3'
    elif row == 4:
        func = '(' + value + ')^(1/3)'
        func_latex = '\\sqrt[3]'+ value_latex
    elif row == 5:
        func = 'ln(' + value + '+1)/ln2'
        func_latex = '\\frac{ln('+ value_latex +'+1)}{ln2}'
    elif row == 6:
        func = '2^' + value + '-1'
        func_latex = '2^' + value_latex + '-1'
    elif row == 7:
        func = '(e^' + value + '-1)/(e-1)'
        func_latex = '\\frac{e^' + value_latex + '-1}{e-1}'

    return row, col, func, func_latex

def beta_index_to_function(index, beta):
    assert len(index) == len(beta)
    function = 'Q = '
    function_latex = '\\boldsymbol{Q}_{score} = '
    for i in range(len(index)):
        row, col, func, func_latex = index2func(index[i])
        if beta[i] < 0:  
            function += f"{beta[i]}*{func}"
        elif beta[i] > 0:
            function += f"+{beta[i]}*{func}"
        function_latex += f"\\beta_{i}{func_latex}+"
        print(f"Index {index[i]} indicates the no.{row+1} func, the no.{col+1} variables, the func expression is {func}") 

    if function[4] == '+':
        function = function[:4] + function[5:]
    function_latex = function_latex[:-1]
    return function, function_latex

