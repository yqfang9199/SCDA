# -*- coding: UTF-8 -*-
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import random
from torch import tensor, float32, save, load
import pickle


class DatasetMDD_SRC(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        with open(data_dir + "NYU_data_175_176_116.pkl", "rb") as fp:
            series = pickle.load(fp)  # 'series' is a list, with each element size: T*N

        sample = len(series)
        numbers = [int(x) for x in range(sample)]
        d1 = zip(numbers, series)
        self.timeseries_dict = dict(d1)
        self.full_subject_list = list(self.timeseries_dict.keys())

        # y is corresponding label
        y = np.load(data_dir + 'NYU_label_175.npy')
        d2 = zip(numbers, y)
        self.behavioral_dict = dict(d2)
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]

    def __len__(self):
        return len(self.full_subject_list)

    def __getitem__(self, idx):
        subject = self.full_subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)
        label = self.behavioral_dict[int(subject)]

        if label == 0:
            label = tensor(0)
        elif label == 1:
            label = tensor(1)
        else:
            raise

        return {'id': subject, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}


class DatasetMDD_TGT(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        with open(data_dir + "UM_data_140_296_116.pkl", "rb") as fp:
            series = pickle.load(fp)  # 'series' is a list, with each element size: T*N

        sample = len(series)
        numbers = [int(x) for x in range(sample)]
        d1 = zip(numbers, series)
        self.timeseries_dict = dict(d1)
        self.full_subject_list = list(self.timeseries_dict.keys())

        # y is corresponding label
        y = np.load(data_dir + 'UM_label_140.npy')
        d2 = zip(numbers, y)
        self.behavioral_dict = dict(d2)
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]

    def __len__(self):
        return len(self.full_subject_list)

    def __getitem__(self, idx):
        subject = self.full_subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)
        label = self.behavioral_dict[int(subject)]

        if label == 0:
            label = tensor(0)
        elif label == 1:
            label = tensor(1)
        else:
            raise

        return {'id': subject, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}
