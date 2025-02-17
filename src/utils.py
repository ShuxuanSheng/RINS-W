import torch
import os
import pickle
import yaml
import numpy as np

# * 表示该函数接受可变数量的参数。传入的参数将会作为一个元组 f_names，其中包含传递给函数的所有参数。
def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names) #将所有传入的文件路径片段用操作系统的路径分隔符组合成一个完整的路径
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict

def pdump(pickle_dict, *f_names): #*f_names是不定参数（通过单星号 * 收集多个位置参数），会被传递为一个包含多个字符串的元组
    """Pickle dump"""
    f_name = os.path.join(*f_names) #将*f_names组合成一个路径
    with open(f_name, "wb") as f:
        pickle.dump(pickle_dict, f)

def mkdir(*paths):
    '''Create a directory if not existing.'''
    path = os.path.join(*paths)
    if not os.path.exists(path):
        os.mkdir(path)

def yload(*f_names):
    """YAML load"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'r') as f:
        yaml_dict = yaml.load(f)
    return yaml_dict

def ydump(yaml_dict, *f_names):
    """YAML dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

def bmv(mat, vec):
    """batch matrix vector product"""
    return torch.einsum('bij, bj -> bi', mat, vec)

def bbmv(mat, vec):
    """double batch matrix vector product"""
    return torch.einsum('baij, baj -> bai', mat, vec)

def bmtv(mat, vec):
    """batch matrix transpose vector product"""
    return torch.einsum('bji, bj -> bi', mat, vec)

def bmtm(mat1, mat2):
    """batch matrix transpose matrix product"""
    return torch.einsum("bji, bjk -> bik", mat1, mat2)

def bmmt(mat1, mat2):
    """batch matrix matrix transpose product"""
    return torch.einsum("bij, bkj -> bik", mat1, mat2)

def bouter(vec1, vec2):
    """batch outer product"""
    return torch.einsum('bi, bj -> bij', vec1, vec2)

def btrace(mat):
    """batch matrix trace"""
    return torch.einsum('bii -> b', mat)

def axat(A, X):
    r"""Returns the product A X A^T."""
    return torch.einsum("ij, jk, lk->il", A, X, A)

