import os
import torch
import numpy as np
from curd import sort

def load_parameter(file_path='model_params.pt'):
    params = torch.load(file_path)
    index = params["index"]
    coef = params["coef"]    
    return index, coef

def save_parameter(index, coef, file_path='model_params.pt', show = True):
    params = {'index': index, 'coef': coef}
    torch.save(params, file_path)
    if show:
        print(f"Parameters saved to {file_path}")

def save_matrix(matrix, file_path='matrix.pt', show = True):
    torch.save(matrix, file_path)
    if show:
        print(f"Matrix saved to {file_path}")

def load_matrix(file_path='matrix.pt'):
    matrix = torch.load(file_path)
    return matrix

def save_logs(logs_matrix, file_name, save_num, sort_num = 9):
    logs = sort(np.array(logs_matrix, dtype=float), order="descending", row=sort_num)[:save_num, :]
    np.savetxt(file_name, logs, fmt=['%f']*(sort_num+1) + ['%d']*1, delimiter=' ')

def create_directory(path):
    if not os.path.exists(path): 
        os.makedirs(path)