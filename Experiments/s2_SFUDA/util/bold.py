import torch
import numpy as np
from random import randrange


def process_dynamic_fc(timeseries, window_size, window_stride, dynamic_length=None, sampling_init=None, self_loop=True):
    # timeseries.shape (batch_size, timepoints, 116)
    # output.shape (batch_size, segment_num, 116, 116)
    if dynamic_length is None:
        dynamic_length = timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert timeseries.shape[1] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    assert timeseries.ndim == 3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(timeseries.shape[1] - dynamic_length + 1)
    sampling_points = list(range(sampling_init, sampling_init + dynamic_length - window_size, window_stride))   # (0, timepoints-window=135, 3)

    dynamic_fc_list = []
    for i in sampling_points:
        fc_list = []
        for _t in timeseries:
            fc = corrcoef(_t[i:i + window_size].T)
            if not self_loop:
                fc -= torch.eye(fc.shape[0])
            fc_list.append(fc)
        dynamic_fc_list.append(torch.stack(fc_list))  # dynamic_fc_list is a list, with each element (batch_size, 116, 116)
    # 'torch.stack(dynamic_fc_list, dim=1)'.shape: (batch_size, segment_num, 116, 116), segment_num (here 45 = timepoints-window_size/window_stride) means the number of sliding windows
    # 'sampling_points': [0， 3， 6， 9 ...], i.e., start point of each window
    return torch.stack(dynamic_fc_list, dim=1), sampling_points


# corrcoef based on https://github.com/pytorch/pytorch/issues/1254
def corrcoef(x):  # x.shape (116, window_size)
    mean_x = torch.mean(x, 1, keepdim=True)  # mean_x.shape [116, 1]
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c
