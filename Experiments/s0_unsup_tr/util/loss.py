import torch
import numpy as np
import torch.nn.functional as F

import numpy as np
import torch.nn as nn
from einops import rearrange, repeat
import logging
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations as comb



def linear(f_of_X, f_of_Y):  # shape: [bs, feat_channel]
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def unitnorm(tensor, dim=None):
    norm = torch.linalg.norm(tensor, dim=dim, keepdims=True)
    return torch.where(norm == 0, tensor, tensor / norm)


def cosine(tensor_1, tensor_2):
    cosine_sim = F.cosine_similarity(tensor_1, tensor_2)
    loss = 1.0 - cosine_sim.mean()
    return loss


def crosscorr_chatgpt(tensor_1, tensor_2):
    # Compute mean and standard deviation of tensor_1 and tensor_2
    mean1, std1 = tensor_1.mean(dim=0), tensor_1.std(dim=0)
    mean2, std2 = tensor_2.mean(dim=0), tensor_2.std(dim=0)

    # Normalize tensor_1 and tensor_2 to have zero mean and unit standard deviation
    tensor_1 = (tensor_1 - mean1) / std1
    tensor_2 = (tensor_2 - mean2) / std2

    # Compute the cross-correlation matrix
    corr = torch.mm(tensor_1.T, tensor_2)

    # Compute the diagonal and off-diagonal elements of the cross-correlation matrix
    on_diag = torch.diag(corr).mean()
    off_diag = (corr.sum() - on_diag) / (corr.numel() - corr.size(0))

    # Maximize the diagonal elements and minimize the off-diagonal elements
    loss = -on_diag + 3.9e-3 * off_diag
    return loss


def crosscorr_norm1(tensor_1, tensor_2):
    # Normalize tensor_1 and tensor_2

    # method - 1 (z-score normalization)
    tensor_1_norm = (tensor_1 - torch.mean(tensor_1, dim=1, keepdim=True)) / torch.std(tensor_1, dim=1, keepdim=True)
    tensor_2_norm = (tensor_2 - torch.mean(tensor_2, dim=1, keepdim=True)) / torch.std(tensor_2, dim=1, keepdim=True)

    # method - 2 (without divide std)
    # tensor_1_norm = tensor_1 - torch.mean(tensor_1, dim=1, keepdim=True)
    # tensor_2_norm = tensor_2 - torch.mean(tensor_2, dim=1, keepdim=True)

    # Compute the cross-correlation matrix
    corr = torch.mm(tensor_1_norm.T, tensor_2_norm)
    corr.div_(tensor_1_norm.shape[0])

    # Compute the diagonal and off-diagonal elements of the cross-correlation matrix
    on_diag = torch.diagonal(corr).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(corr).pow_(2).sum()

    # Maximize the diagonal elements and minimize the off-diagonal elements
    loss = on_diag + 3.9e-3 * off_diag
    return loss


def crosscorr_norm2(tensor_1, tensor_2):
    # Normalize tensor_1 and tensor_2

    # method - 1 (z-score normalization)
    # tensor_1_norm = (tensor_1 - torch.mean(tensor_1, dim=1, keepdim=True)) / torch.std(tensor_1, dim=1, keepdim=True)
    # tensor_2_norm = (tensor_2 - torch.mean(tensor_2, dim=1, keepdim=True)) / torch.std(tensor_2, dim=1, keepdim=True)

    # method - 2 (without divide std)
    tensor_1_norm = tensor_1 - torch.mean(tensor_1, dim=1, keepdim=True)
    tensor_2_norm = tensor_2 - torch.mean(tensor_2, dim=1, keepdim=True)

    # Compute the cross-correlation matrix
    corr = torch.mm(tensor_1_norm.T, tensor_2_norm)
    corr.div_(tensor_1_norm.shape[0])

    # Compute the diagonal and off-diagonal elements of the cross-correlation matrix
    on_diag = torch.diagonal(corr).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(corr).pow_(2).sum()

    # Maximize the diagonal elements and minimize the off-diagonal elements
    loss = on_diag + 3.9e-3 * off_diag
    return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vari_loss(tensor_1, tensor_2):

    tensor_1_std = torch.sqrt(tensor_1.var(dim=0) + 0.0001)
    tensor_2_std = torch.sqrt(tensor_2.var(dim=0) + 0.0001)

    variance_loss = torch.mean(F.relu(1 - tensor_1_std)) / 2 + torch.mean(F.relu(1 - tensor_2_std)) / 2
    return variance_loss



# tensor_1 = torch.tensor([[1.0, 2.0, 2.0], [2.0, 3.0, 4.0]])
# tensor_2 = torch.tensor([[1.0, 2.0, 2.0], [2.0, 3.0, 4.0]])
# vari_loss(tensor_1, tensor_2)
# print('1')
# b = torch.tensor([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
# print(crosscorr(a, b))
# # print('1')

# a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# b = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
# print(cosine(a, b))
# print('1')

# a = torch.tensor([[1.0, 1.0], [1.0, 1.0, 2.0]])
# a = torch.tensor([[0.0, 0.0], [1.0, 1.0, 2.0]])

