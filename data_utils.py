import os
import torch

def save_parameter(index, beta, file_path='model_params.pt'):
    # 定义参数
    # beta = torch.tensor([1, 3, 5])
    # index = torch.tensor([2, 5, 6])
    params = {'index': index, 'beta': beta}
    torch.save(params, file_path)
    print(f"Parameters saved to {file_path}")

def load_parameter(file_path='model_params.pt'):
    # 加载参数
    loaded_params = torch.load(file_path)
    index = loaded_params["index"]
    beta = loaded_params["beta"]    
    return index, beta

def save_matrix(matrix, file_path='matrix.pt'):
    # matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    torch.save(matrix, file_path)
    print(f"Matrix saved to {file_path}")

def load_matrix(file_path='matrix.pt'):
    # 加载矩阵
    loaded_matrix = torch.load(file_path)
    return loaded_matrix

def create_directory(path):
    if not os.path.exists(path): 
        os.makedirs(path)