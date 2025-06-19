import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score

def regression(X, y, index):
    U, S, V = torch.linalg.svd(X[:, index], full_matrices=False)
    inv_X_s = V.transpose(-2, -1) @ torch.diag(1 / S) @ U.transpose(-2, -1)
    coef = inv_X_s @ y
    return coef.squeeze()

def prediction(X, coef, index):
    X_s = X[:, index]
    yhat = X_s @ coef
    return yhat

def regression_ridge(X, y, index):
    # X, y is torch.Tensor, index is a list of indices
    X, y = np.array(X), np.array(y)
    X_s = X[:, index]

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]  # 候选的 alpha 值
    ridge_cv = RidgeCV(alphas=alphas, cv=5)  # 5 折交叉验证
    ridge_cv.fit(X, y)

    reg = Ridge(alpha=ridge_cv.alpha_).fit(X_s, y)
    return reg

# 测试模型
def prediction_ridge(X, reg, index):
    X = np.array(X)
    X_s = X[:, index]
    yhat = reg.predict(X_s)
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

def calculate_sp(y, yhat, show = True):
    plcc, p_PLCC = pearsonr(y, yhat)
    srcc, p_SRCC = spearmanr(y, yhat)

    err_flag = False
    if p_PLCC >0.05:
        if show:
            print("The plcc correlation is not significant.")
        err_flag = True
    if p_SRCC >0.05:
        if show:
            print("The srcc correlation is not significant.")
        err_flag = True
    plcc, srcc = torch.abs(torch.tensor(plcc)).item(), torch.abs(torch.tensor(srcc)).item()
    if show:
        print(f'Testing plcc {plcc},\tsrcc {srcc}.')
    return plcc, srcc, err_flag

def calculate_mse_r2(y, yhat):
    mse = mean_squared_error(y, yhat)
    r2 = r2_score(y, yhat)
    print(f'MSE: {mse}, R^2: {r2}')

def function_to_latex(index, data_dim = 6):
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

def coef_index_to_function(index, coef):
    assert len(index) == len(coef)
    function = 'Q = '
    function_latex = '\\boldsymbol{Q}_{score} = '
    for i in range(len(index)):
        row, col, func, func_latex = function_to_latex(index[i])
        if coef[i] < 0:  
            function += f"{coef[i]}*{func}"
        elif coef[i] > 0:
            function += f"+{coef[i]}*{func}"
        function_latex += f"\\beta_{i}{func_latex}+"
        print(f"Index {index[i]} indicates the no.{row+1} func, the no.{col+1} variables, the func expression is {func}") 

    if function[4] == '+':
        function = function[:4] + function[5:]
    function_latex = function_latex[:-1]
    print(f"final function: {function}, \n{function_latex}")

