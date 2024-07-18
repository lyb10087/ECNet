import torch
from torch import nn


def to_svd(weight, k=90):
    U, sigma, VT = torch.linalg.svd(weight)
    return U[:, :k], sigma[:k], VT[:k, :]

def inverse_svd(U, sigma, VT):
    sigma = torch.diag(sigma)
    result_svd = torch.mm(torch.mm(U, sigma), VT)
    return result_svd

def to_two_dim(ten: torch.Tensor):
    a, b, c, d = ten.shape
    temp = ten.contiguous().view(a * b, c * d)
    return temp


def back_four_dim(ten: torch.Tensor, a, b, c, d):
    return ten.view(a, b, c, d)


def svd_gather(ten: torch.Tensor,K):
    aa, bb, cc, dd = ten.shape
    ten = to_two_dim(ten)
    u, s, v = to_svd(ten,K)
    u_list = [u]
    s_list = [s]
    v_list = [v]

    ans = []
    for i in range(len(u_list)):
        x = inverse_svd(u_list[i], s_list[i], v_list[i])
        x = back_four_dim(x, aa, bb, cc, dd)
        ans.append(x)
    return ans






