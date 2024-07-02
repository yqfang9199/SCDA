import torch
import numpy as np


def linear(f_of_X, f_of_Y):  # shape: [bs, feat_channel]
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss