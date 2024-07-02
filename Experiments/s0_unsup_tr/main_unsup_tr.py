import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.filterwarnings('ignore')

import math
from dataset_prep import Combine_Dataset, DatasetMDD_SRC, DatasetMDD_TGT
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import confusion_matrix
import random
from sklearn.metrics import roc_auc_score
from model import *
import util
import linecache
import re
from scipy.signal import resample
import sys
sys.path.append('./util')
from loss import vari_loss, linear, unitnorm, cosine


def step_decay(epoch, learning_rate, drop, epochs_drop):
    """
    learning rate step decay
    :param epoch: current training epoch
    :param learning_rate: initial learning rate
    :return: learning rate after step decay
    """
    initial_lrate = learning_rate
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def train(epoch, model_m1, model_m2, model_m3, learning_rate, tgt_loader, drop, epochs_drop):
    log_interval = 1
    LEARNING_RATE = step_decay(epoch, learning_rate, drop, epochs_drop)
    print(f'Learning Rate: {LEARNING_RATE}')
    optimizer_m1 = optim.Adam(model_m1.parameters(), lr=LEARNING_RATE)
    optimizer_m2 = optim.Adam(model_m2.parameters(), lr=LEARNING_RATE)
    optimizer_m3 = optim.Adam(model_m3.parameters(), lr=LEARNING_RATE)

    model_m1.train()
    model_m2.train()
    model_m3.train()

    consistency_loss_accumulate = 0.0

    len_dataloader = len(tgt_loader)
    for step, tgt_sample_batch in enumerate(tgt_loader):
        optimizer_m1.zero_grad()
        optimizer_m2.zero_grad()
        optimizer_m3.zero_grad()

        # m1 output ---------------------------------------------------------------------------------------------
        pct_step = random.choice(pct)
        dyn_a_m1, sampling_points_m1 = util.bold.process_dynamic_fc(tgt_sample_batch['timeseries'], window_size=40, window_stride=30, dynamic_length=int(170*pct_step))
        sampling_endpoints_m1 = [p + 40 for p in sampling_points_m1]

        # generation of "dyn_v" is from original github code
        if step == 0:
            dyn_v_m1 = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points_m1), b=BATCH_SIZE)
        if not dyn_v_m1.shape[1] == dyn_a_m1.shape[1]:
            dyn_v_m1 = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points_m1), b=BATCH_SIZE)
        if len(dyn_a_m1) < BATCH_SIZE:
            dyn_v_m1 = dyn_v_m1[:len(dyn_a_m1)]

        t_m1 = tgt_sample_batch['timeseries'].permute(1, 0, 2)  # t.shape [timepoints, batch_size, 116]

        logit_m1, _, feat_G_m1, feat_T_m1, feat_L_m1 = model_m1(dyn_v_m1.to(device), dyn_a_m1.to(device), t_m1.to(device), sampling_endpoints_m1)

        # m2 output ---------------------------------------------------------------------------------------------
        resamp_step = random.choice(resamp)
        dyn_a_m2, sampling_points_m2 = util.bold.process_dynamic_fc(torch.from_numpy(resample(tgt_sample_batch['timeseries'], num=int(170*resamp_step), axis=1)), window_size=40, window_stride=30, dynamic_length=int(170*resamp_step))
        sampling_endpoints_m2 = [p + 40 for p in sampling_points_m2]

        # generation of "dyn_v" is from original github code
        if step == 0:
            dyn_v_m2 = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points_m2), b=BATCH_SIZE)
        if not dyn_v_m2.shape[1] == dyn_a_m2.shape[1]:
            dyn_v_m2 = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points_m2), b=BATCH_SIZE)
        if len(dyn_a_m2) < BATCH_SIZE:
            dyn_v_m2 = dyn_v_m2[:len(dyn_a_m2)]

        t_m2 = tgt_sample_batch['timeseries'].permute(1, 0, 2)  # t.shape [timepoints, batch_size, 116]

        logit_m2, _, feat_G_m2, feat_T_m2, feat_L_m2 = model_m2(dyn_v_m2.to(device), dyn_a_m2.to(device), t_m2.to(device), sampling_endpoints_m2)

        # m3 output ---------------------------------------------------------------------------------------------
        winsize_step = random.choice(winsize)
        dyn_a_m3, sampling_points_m3 = util.bold.process_dynamic_fc(tgt_sample_batch['timeseries'], window_size=winsize_step, window_stride=30, dynamic_length=170)
        sampling_endpoints_m3 = [p + 40 for p in sampling_points_m3]

        # generation of "dyn_v" is from original github code
        if step == 0:
            dyn_v_m3 = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points_m3), b=BATCH_SIZE)
        if not dyn_v_m3.shape[1] == dyn_a_m3.shape[1]:
            dyn_v_m3 = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points_m3), b=BATCH_SIZE)
        if len(dyn_a_m3) < BATCH_SIZE:
            dyn_v_m3 = dyn_v_m3[:len(dyn_a_m3)]

        t_m3 = tgt_sample_batch['timeseries'].permute(1, 0, 2)  # t.shape [timepoints, batch_size, 116]

        logit_m3, _, feat_G_m3, feat_T_m3, feat_L_m3 = model_m3(dyn_v_m3.to(device), dyn_a_m3.to(device), t_m3.to(device), sampling_endpoints_m3)

        # consistency loss
        consistency_loss_T_cosine = 1.0 * cosine(feat_T_m1[:, 0, :], feat_T_m2[:, 0, :]) + 1.0 * cosine(feat_T_m1[:, 1, :], feat_T_m2[:, 1, :]) + \
                                    1.0 * cosine(feat_T_m1[:, 0, :], feat_T_m3[:, 0, :]) + 1.0 * cosine(feat_T_m1[:, 1, :], feat_T_m3[:, 1, :]) + \
                                    1.0 * cosine(feat_T_m2[:, 0, :], feat_T_m3[:, 0, :]) + 1.0 * cosine(feat_T_m2[:, 1, :], feat_T_m3[:, 1, :])
        consistency_loss_T_var = 0.01 * vari_loss(feat_T_m1[:, 0, :], feat_T_m2[:, 0, :]) + 0.01 * vari_loss(feat_T_m1[:, 1, :], feat_T_m2[:, 1, :]) + \
                                 0.01 * vari_loss(feat_T_m1[:, 0, :], feat_T_m3[:, 0, :]) + 0.01 * vari_loss(feat_T_m1[:, 1, :], feat_T_m3[:, 1, :]) + \
                                 0.01 * vari_loss(feat_T_m2[:, 0, :], feat_T_m3[:, 0, :]) + 0.01 * vari_loss(feat_T_m2[:, 1, :], feat_T_m3[:, 1, :])
        consistency_loss_T = consistency_loss_T_cosine + consistency_loss_T_var

        consistency_loss_L_cosine = 1.0 * cosine(feat_L_m1[:, 0, :], feat_L_m2[:, 0, :]) + 1.0 * cosine(feat_L_m1[:, 1, :], feat_L_m2[:, 1, :]) + \
                                    1.0 * cosine(feat_L_m1[:, 0, :], feat_L_m3[:, 0, :]) + 1.0 * cosine(feat_L_m1[:, 1, :], feat_L_m3[:, 1, :]) + \
                                    1.0 * cosine(feat_L_m2[:, 0, :], feat_L_m3[:, 0, :]) + 1.0 * cosine(feat_L_m2[:, 1, :], feat_L_m3[:, 1, :])
        consistency_loss_L_var = 0.01 * vari_loss(feat_L_m1[:, 0, :], feat_L_m2[:, 0, :]) + 0.01 * vari_loss(feat_L_m1[:, 1, :], feat_L_m2[:, 1, :]) + \
                                 0.01 * vari_loss(feat_L_m1[:, 0, :], feat_L_m3[:, 0, :]) + 0.01 * vari_loss(feat_L_m1[:, 1, :], feat_L_m3[:, 1, :]) + \
                                 0.01 * vari_loss(feat_L_m2[:, 0, :], feat_L_m3[:, 0, :]) + 0.01 * vari_loss(feat_L_m2[:, 1, :], feat_L_m3[:, 1, :])
        consistency_loss_L = consistency_loss_L_cosine + consistency_loss_L_var

        consistency_loss = consistency_loss_T + consistency_loss_L

        consistency_loss.backward()
        optimizer_m1.step()
        optimizer_m2.step()
        optimizer_m3.step()

        consistency_loss_accumulate += consistency_loss

        if (step + 1) % log_interval == 0:
            print("Train Epoch [{:4d}/{}] Step [{:2d}/{}]: consistency_loss_T={:.6f} consistency_loss_L={:.6f}".format(epoch, TRAIN_EPOCHS, step + 1, len_dataloader, consistency_loss_T.data, consistency_loss_L.data))

    consistency_loss_accumulate /= len_dataloader

    print('Consistency Loss: {:.4f}'.format(consistency_loss_accumulate))

    # save checkpoint.pth, save train loss and acc to a txt file
    if epoch % 5 == 0:
        model_avg = ModelSTAGIN(input_dim=116, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.5, cls_token='sum', readout='sero')
        for param1, param2, param3, param_avg in zip(model_m1.parameters(), model_m2.parameters(), model_m3.parameters(), model_avg.parameters()):
            param_avg.data.copy_((param1.data + param2.data + param3.data) / 3.0)
        torch.save(model_avg.state_dict(), SAVE_PATH + 'unsuptr_mavg_fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth')

    with open(SAVE_PATH + 'fold_' + str(fold) + '_train_loss_and_acc.txt', 'a') as f:
        f.write('epoch {}, Consistency Loss {:.5}\n'.format(epoch, consistency_loss_accumulate))


if __name__ == '__main__':

    ROOT_PATH = '../Data/'
    SAVE_PATH = '../Experiments/s0_unsup_tr/'

    BATCH_SIZE = 900
    TRAIN_EPOCHS = 5
    learning_rate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    FOLD_NUM = 1

    pct = [0.85, 0.90, 0.95, 1.0]
    resamp = [1.0, 1.1, 1.3, 1.0, 1/1.1, 1/1.3]
    winsize = [40, 60, 80, 100]

    for fold in range(FOLD_NUM):  # repeat multiple times
        print('fold:', fold)
        seed = fold  # fold
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致

        target_dataset = Combine_Dataset(ROOT_PATH)
        target_train_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

        print('Construct model:')
        ddcnet_m1 = ModelSTAGIN(input_dim=116, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.5, cls_token='sum', readout='sero')
        ddcnet_m2 = ModelSTAGIN(input_dim=116, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.5, cls_token='sum', readout='sero')
        ddcnet_m3 = ModelSTAGIN(input_dim=116, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.5, cls_token='sum', readout='sero')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device:', device)

        ddcnet_m1.to(device)
        ddcnet_m2.to(device)
        ddcnet_m3.to(device)

        with open(SAVE_PATH + 'fold_' + str(fold) + '_train_loss_and_acc.txt', 'a') as f:
            f.write('total_epoch: {}, batch_size: {}, initial_lr {:.8}, drop_lr: {:.5}, drop_lr_per_epoch\n'.format(TRAIN_EPOCHS, BATCH_SIZE, learning_rate, drop, epochs_drop))

        for epoch in range(1, TRAIN_EPOCHS + 1):
            print(f'Train Epoch {epoch}:')
            train(epoch, ddcnet_m1, ddcnet_m2, ddcnet_m3, learning_rate, target_train_loader, drop, epochs_drop)
