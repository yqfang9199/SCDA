import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.filterwarnings('ignore')

import math
from dataset_prep import DatasetMDD_TGT  # DatasetMDD_SRC
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
        dyn_a_m1, sampling_points_m1 = util.bold.process_dynamic_fc(tgt_sample_batch['timeseries'], window_size=40, window_stride=30, dynamic_length=int(296*pct_step))
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
        dyn_a_m2, sampling_points_m2 = util.bold.process_dynamic_fc(torch.from_numpy(resample(tgt_sample_batch['timeseries'], num=int(296*resamp_step), axis=1)), window_size=40, window_stride=30, dynamic_length=int(296*resamp_step))
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
        dyn_a_m3, sampling_points_m3 = util.bold.process_dynamic_fc(tgt_sample_batch['timeseries'], window_size=winsize_step, window_stride=30, dynamic_length=296)
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
        consistency_loss_T = 1.0 * util.loss.linear(feat_T_m1[:, 0, :], feat_T_m2[:, 0, :]) + 1.0 * util.loss.linear(feat_T_m1[:, 1, :], feat_T_m2[:, 1, :]) + \
                             1.0 * util.loss.linear(feat_T_m1[:, 0, :], feat_T_m3[:, 0, :]) + 1.0 * util.loss.linear(feat_T_m1[:, 1, :], feat_T_m3[:, 1, :]) + \
                             1.0 * util.loss.linear(feat_T_m2[:, 0, :], feat_T_m3[:, 0, :]) + 1.0 * util.loss.linear(feat_T_m2[:, 1, :], feat_T_m3[:, 1, :])

        consistency_loss_L = 1.0 * util.loss.linear(feat_L_m1[:, 0, :], feat_L_m2[:, 0, :]) + 1.0 * util.loss.linear(feat_L_m1[:, 1, :], feat_L_m2[:, 1, :]) + \
                             1.0 * util.loss.linear(feat_L_m1[:, 0, :], feat_L_m3[:, 0, :]) + 1.0 * util.loss.linear(feat_L_m1[:, 1, :], feat_L_m3[:, 1, :]) + \
                             1.0 * util.loss.linear(feat_L_m2[:, 0, :], feat_L_m3[:, 0, :]) + 1.0 * util.loss.linear(feat_L_m2[:, 1, :], feat_L_m3[:, 1, :])

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
    if epoch == 150:
        torch.save(model_m1.state_dict(), SAVE_PATH + 'm1_fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth')
        torch.save(model_m2.state_dict(), SAVE_PATH + 'm2_fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth')
        torch.save(model_m3.state_dict(), SAVE_PATH + 'm3_fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth')

    with open(SAVE_PATH + 's0epochid_' + str(s0epochid) + '_s1saveepoch_' + str(s1saveepoch) + '_fold_' + str(fold) + '_train_loss_and_acc.txt', 'a') as f:
        f.write('epoch {}, Consistency Loss {:.5}\n'.format(epoch, consistency_loss_accumulate))


def test_m1(model_m1, target_loader) -> object:

    with torch.no_grad():
        model_m1.eval()
        test_correct = 0
        TN = 0
        FP = 0
        FN = 0
        TP = 0

        te_auc_y_gt = []
        te_auc_y_probs = []
        te_acc_y_pred = []

        for step, test_sample_batch in enumerate(target_loader):
            dyn_a, sampling_points = util.bold.process_dynamic_fc(test_sample_batch['timeseries'], window_size=40, window_stride=30, dynamic_length=296)
            sampling_endpoints = [p + 40 for p in sampling_points]

            # generation of "dyn_v" is from original github code
            if step == 0:
                dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=BATCH_SIZE)
            if not dyn_v.shape[1] == dyn_a.shape[1]:
                dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=BATCH_SIZE)
            if len(dyn_a) < BATCH_SIZE:
                dyn_v = dyn_v[:len(dyn_a)]

            t = test_sample_batch['timeseries'].permute(1, 0, 2)
            label = test_sample_batch['label']

            logit, _, _, _, _ = model_m1(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)  # dyn_v, dyn_a: (batch_size, segment_num, 116, 116)

            test_pred = logit.argmax(1)
            test_prob = logit.softmax(1)[:, 1]  # prob

            test_correct += sum(test_pred.data.cpu().numpy() == label.cpu().numpy())
            TN_tmp, FP_tmp, FN_tmp, TP_tmp = confusion_matrix(label.cpu().numpy(), test_pred.data.cpu().numpy(), labels=[0, 1]).ravel()
            TN += TN_tmp
            FP += FP_tmp
            FN += FN_tmp
            TP += TP_tmp
            te_auc_y_gt.extend(label.cpu().numpy())
            te_auc_y_probs.extend(test_prob.detach().cpu().numpy())
            te_acc_y_pred.extend(test_pred.detach().cpu().numpy())

        TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate
        TNR = TN / (TN + FP)  # Specificity/ true negative rate
        PPV = TP / (TP + FP)  # Precision/ positive predictive value
        test_acc = (TP+TN) / (TP + FP + TN + FN)  # accuracy of each class
        test_AUC = roc_auc_score(te_auc_y_gt, te_auc_y_probs)
        test_F1 = (2 * TP) / (2 * TP + FP + FN)

        print('Test set: Correct_num: {}, test_acc: {:.4f}, test_AUC: {:.4f}, test_F1: {:.4f}, TPR: {:.4f}, TNR: {:.4f}, PPV:{:.4f}\n'.format(
            test_correct, test_acc, test_AUC, test_F1, TPR, TNR, PPV))

        # save test loss and acc to a txt file
        # with open(SAVE_PATH + 's0epochid_' + str(s0epochid) + '_s1saveepoch_' + str(s1saveepoch) + '_m1_fold_' + str(fold) + '_test_loss_and_acc.txt', 'a') as f:
        #     f.write('epoch {}, test_acc {:.5}, test_AUC {:.5}, test_F1: {:.4f}, TPR {:.5}, TNR {:.5}, PPV {:.5}\n'.format(epoch, test_acc, test_AUC, test_F1, TPR, TNR, PPV))

        return te_auc_y_gt, te_acc_y_pred, te_auc_y_probs


def test_m2(model_m2, target_loader) -> object:

    with torch.no_grad():
        model_m2.eval()
        test_correct = 0
        TN = 0
        FP = 0
        FN = 0
        TP = 0

        te_auc_y_gt = []
        te_auc_y_probs = []
        te_acc_y_pred = []

        for step, test_sample_batch in enumerate(target_loader):
            dyn_a, sampling_points = util.bold.process_dynamic_fc(test_sample_batch['timeseries'], window_size=40, window_stride=30, dynamic_length=296)
            sampling_endpoints = [p + 40 for p in sampling_points]

            # generation of "dyn_v" is from original github code
            if step == 0:
                dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=BATCH_SIZE)
            if not dyn_v.shape[1] == dyn_a.shape[1]:
                dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=BATCH_SIZE)
            if len(dyn_a) < BATCH_SIZE:
                dyn_v = dyn_v[:len(dyn_a)]

            t = test_sample_batch['timeseries'].permute(1, 0, 2)
            label = test_sample_batch['label']

            logit, _, _, _, _ = model_m2(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)  # dyn_v, dyn_a: (batch_size, segment_num, 116, 116)

            test_pred = logit.argmax(1)
            test_prob = logit.softmax(1)[:, 1]  # prob

            test_correct += sum(test_pred.data.cpu().numpy() == label.cpu().numpy())
            TN_tmp, FP_tmp, FN_tmp, TP_tmp = confusion_matrix(label.cpu().numpy(), test_pred.data.cpu().numpy(), labels=[0, 1]).ravel()
            TN += TN_tmp
            FP += FP_tmp
            FN += FN_tmp
            TP += TP_tmp
            te_auc_y_gt.extend(label.cpu().numpy())
            te_auc_y_probs.extend(test_prob.detach().cpu().numpy())
            te_acc_y_pred.extend(test_pred.detach().cpu().numpy())

        TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate
        TNR = TN / (TN + FP)  # Specificity/ true negative rate
        PPV = TP / (TP + FP)  # Precision/ positive predictive value
        test_acc = (TP+TN) / (TP + FP + TN + FN)  # accuracy of each class
        test_AUC = roc_auc_score(te_auc_y_gt, te_auc_y_probs)
        test_F1 = (2 * TP) / (2 * TP + FP + FN)

        print('Test set: Correct_num: {}, test_acc: {:.4f}, test_AUC: {:.4f}, test_F1: {:.4f}, TPR: {:.4f}, TNR: {:.4f}, PPV:{:.4f}\n'.format(
            test_correct, test_acc, test_AUC, test_F1, TPR, TNR, PPV))

        # save test loss and acc to a txt file
        # with open(SAVE_PATH + 's0epochid_' + str(s0epochid) + '_s1saveepoch_' + str(s1saveepoch) + '_m2_fold_' + str(fold) + '_test_loss_and_acc.txt', 'a') as f:
        #     f.write('epoch {}, test_acc {:.5}, test_AUC {:.5}, test_F1: {:.4f}, TPR {:.5}, TNR {:.5}, PPV {:.5}\n'.format(epoch, test_acc, test_AUC, test_F1, TPR, TNR, PPV))

        return te_auc_y_gt, te_acc_y_pred, te_auc_y_probs


def test_m3(model_m3, target_loader) -> object:

    with torch.no_grad():
        model_m3.eval()
        test_correct = 0
        TN = 0
        FP = 0
        FN = 0
        TP = 0

        te_auc_y_gt = []
        te_auc_y_probs = []
        te_acc_y_pred = []

        for step, test_sample_batch in enumerate(target_loader):
            dyn_a, sampling_points = util.bold.process_dynamic_fc(test_sample_batch['timeseries'], window_size=40, window_stride=30, dynamic_length=296)
            sampling_endpoints = [p + 40 for p in sampling_points]

            # generation of "dyn_v" is from original github code
            if step == 0:
                dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=BATCH_SIZE)
            if not dyn_v.shape[1] == dyn_a.shape[1]:
                dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=BATCH_SIZE)
            if len(dyn_a) < BATCH_SIZE:
                dyn_v = dyn_v[:len(dyn_a)]

            t = test_sample_batch['timeseries'].permute(1, 0, 2)
            label = test_sample_batch['label']

            logit, _, _, _, _ = model_m3(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)  # dyn_v, dyn_a: (batch_size, segment_num, 116, 116)

            test_pred = logit.argmax(1)
            test_prob = logit.softmax(1)[:, 1]  # prob

            test_correct += sum(test_pred.data.cpu().numpy() == label.cpu().numpy())
            TN_tmp, FP_tmp, FN_tmp, TP_tmp = confusion_matrix(label.cpu().numpy(), test_pred.data.cpu().numpy(), labels=[0, 1]).ravel()
            TN += TN_tmp
            FP += FP_tmp
            FN += FN_tmp
            TP += TP_tmp
            te_auc_y_gt.extend(label.cpu().numpy())
            te_auc_y_probs.extend(test_prob.detach().cpu().numpy())
            te_acc_y_pred.extend(test_pred.detach().cpu().numpy())

        TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate
        TNR = TN / (TN + FP)  # Specificity/ true negative rate
        PPV = TP / (TP + FP)  # Precision/ positive predictive value
        test_acc = (TP+TN) / (TP + FP + TN + FN)  # accuracy of each class
        test_AUC = roc_auc_score(te_auc_y_gt, te_auc_y_probs)
        test_F1 = (2 * TP) / (2 * TP + FP + FN)

        print('Test set: Correct_num: {}, test_acc: {:.4f}, test_AUC: {:.4f}, test_F1: {:.4f}, TPR: {:.4f}, TNR: {:.4f}, PPV:{:.4f}\n'.format(
            test_correct, test_acc, test_AUC, test_F1, TPR, TNR, PPV))

        # save test loss and acc to a txt file
        # with open(SAVE_PATH + 's0epochid_' + str(s0epochid) + '_s1saveepoch_' + str(s1saveepoch) + '_m3_fold_' + str(fold) + '_test_loss_and_acc.txt', 'a') as f:
        #     f.write('epoch {}, test_acc {:.5}, test_AUC {:.5}, test_F1: {:.4f}, TPR {:.5}, TNR {:.5}, PPV {:.5}\n'.format(epoch, test_acc, test_AUC, test_F1, TPR, TNR, PPV))

        return te_auc_y_gt, te_acc_y_pred, te_auc_y_probs


def test_avg3br() -> object:
    gt = m1_gt
    probs_avg3br = np.mean([np.array(m1_probs), np.array(m2_probs), np.array(m3_probs)], axis=0)
    preb_avg3br = probs_avg3br >= 0.3

    TN, FP, FN, TP = confusion_matrix(np.array(gt), np.array(preb_avg3br), labels=[0, 1]).ravel()
    TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate
    TNR = TN / (TN + FP)  # Specificity/ true negative rate
    PPV = TP / (TP + FP)  # Precision/ positive predictive value
    test_acc = (TP+TN) / (TP + FP + TN + FN)  # accuracy of each class
    test_F1 = (2 * TP) / (2 * TP + FP + FN)
    test_AUC = roc_auc_score(gt, probs_avg3br)

    print('Test set: test_acc: {:.4f}, test_AUC: {:.4f}, test_F1: {:.4f}, TPR: {:.4f}, TNR: {:.4f}, PPV:{:.4f}\n'.format(
        test_acc, test_AUC, test_F1, TPR, TNR, PPV))

    with open(SAVE_PATH + 's0epochid_' + str(s0epochid) + '_s1saveepoch_' + str(s1saveepoch) + '_avg3br_fold_' + str(fold) + '_test_loss_and_acc.txt', 'a') as f:
        f.write('epoch {}, test_acc {:.5}, test_AUC: {:.4f}, test_F1: {:.4f}, TPR {:.5}, TNR {:.5}, PPV {:.5}\n'.format(epoch, test_acc, test_AUC, test_F1, TPR, TNR, PPV))


if __name__ == '__main__':

    ROOT_PATH = '../Data/'
    SAVE_PATH = '../Experiments/s2_SFUDA/checkpoint_SFUDA_avg3branches/'

    BATCH_SIZE = 140
    TRAIN_EPOCHS = 150
    learning_rate = 0.0003
    drop = 0.5
    epochs_drop = 50.0
    FOLD_NUM = 5

    pct = [0.85, 0.90, 0.95, 1.0]
    resamp = [1.0, 1.1, 1.3, 1.0, 1/1.1, 1/1.3]
    winsize = [40, 60, 80, 100]

    for s0epochid in [5]:  # so_unsuptr [5, 10, 15]
        for s1saveepoch in [50]:  # s1_pretrain [50, 100, 150, 200]
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

                target_dataset = DatasetMDD_TGT(ROOT_PATH)
                target_train_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
                target_test_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

                print('Construct model begin:')
                print('Load pretrained model:')
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # print('device:', device)
                pre_ddcnet = ModelSTAGIN(input_dim=116, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.5, cls_token='sum', readout='sero')
                pre_ddcnet.load_state_dict(torch.load('../Experiments/s1_pretrain/checkpoints_pretrain/mavg_' + str(s0epochid) + '_fold_' + str(fold) + '_epoch_' + str(s1saveepoch) + '.pth', map_location=device))

                print('Construct model1:')
                ddcnet_pre_dict_m1 = pre_ddcnet.state_dict()
                ddcnet_m1 = ModelSTAGIN(input_dim=116, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.5, cls_token='sum', readout='sero')
                ddcnet_dict_m1 = ddcnet_m1.state_dict()
                ddcnet_pre_dict_m1 = {k: v for k, v in ddcnet_pre_dict_m1.items() if k in ddcnet_dict_m1}
                ddcnet_dict_m1.update(ddcnet_pre_dict_m1)
                ddcnet_m1.load_state_dict(ddcnet_dict_m1)

                print('Construct model2:')
                ddcnet_pre_dict_m2 = pre_ddcnet.state_dict()
                ddcnet_m2 = ModelSTAGIN(input_dim=116, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.5, cls_token='sum', readout='sero')
                ddcnet_dict_m2 = ddcnet_m2.state_dict()
                ddcnet_pre_dict_m2 = {k: v for k, v in ddcnet_pre_dict_m2.items() if k in ddcnet_dict_m2}
                ddcnet_dict_m2.update(ddcnet_pre_dict_m2)
                ddcnet_m2.load_state_dict(ddcnet_dict_m2)

                print('Construct model3:')
                ddcnet_pre_dict_m3 = pre_ddcnet.state_dict()
                ddcnet_m3 = ModelSTAGIN(input_dim=116, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.5, cls_token='sum', readout='sero')
                ddcnet_dict_m3 = ddcnet_m3.state_dict()
                ddcnet_pre_dict_m3 = {k: v for k, v in ddcnet_pre_dict_m3.items() if k in ddcnet_dict_m3}
                ddcnet_dict_m3.update(ddcnet_pre_dict_m3)
                ddcnet_m3.load_state_dict(ddcnet_dict_m3)


                ddcnet_m1.to(device)
                ddcnet_m2.to(device)
                ddcnet_m3.to(device)

                with open(SAVE_PATH + 's0epochid_' + str(s0epochid) + '_s1saveepoch_' + str(s1saveepoch) + '_fold_' + str(fold) + '_train_loss_and_acc.txt', 'a') as f:
                    f.write('total_epoch: {}, batch_size: {}, initial_lr {:.8}, drop_lr: {:.5}, drop_lr_per_epoch\n'.format(TRAIN_EPOCHS, BATCH_SIZE, learning_rate, drop, epochs_drop))
                with open(SAVE_PATH + 's0epochid_' + str(s0epochid) + '_s1saveepoch_' + str(s1saveepoch) + '_avg3br_fold_' + str(fold) + '_test_loss_and_acc.txt', 'a') as f:
                    f.write('total_epoch: {}, batch_size: {}, initial_lr {:.8}, drop_lr: {:.5}, drop_lr_per_epoch\n'.format(TRAIN_EPOCHS, BATCH_SIZE, learning_rate, drop, epochs_drop))

                for epoch in range(1, TRAIN_EPOCHS + 1):
                    print(f'Train Epoch {epoch}:')
                    train(epoch, ddcnet_m1, ddcnet_m2, ddcnet_m3, learning_rate, target_train_loader, drop, epochs_drop)
                    m1_gt, m1_pred, m1_probs = test_m1(ddcnet_m1, target_test_loader)
                    m2_gt, m2_pred, m2_probs = test_m2(ddcnet_m2, target_test_loader)
                    m3_gt, m3_pred, m3_probs = test_m3(ddcnet_m3, target_test_loader)
                    avg3br_prediction = test_avg3br()

            # calculate the mean value of five folds
            def get_line_context(file_path, line_number=(TRAIN_EPOCHS+1)):
                return linecache.getline(file_path, line_number).strip()


            # average 5 folds for avg3br ---------------------------------------------------------------------------
            for line_num in range(2, TRAIN_EPOCHS + 2):
                test_result_list = []
                for fold in range(FOLD_NUM):
                    txt_file_path = SAVE_PATH + 's0epochid_' + str(s0epochid) + '_s1saveepoch_' + str(s1saveepoch) + '_avg3br_fold_' + str(fold) + '_test_loss_and_acc.txt'
                    test_result_str = get_line_context(txt_file_path, line_number=line_num)
                    test_result_str = test_result_str.replace('nan', '10000')
                    test_result_str_numpart = re.findall(r"\d+\.?\d*", test_result_str)  # only extract number in a str
                    test_result_str_numpart_float = []
                    for num in test_result_str_numpart:
                        test_result_str_numpart_float.append(float(num))

                    test_result_list.append(test_result_str_numpart_float)

                test_acc_list = []
                test_auc_list = []
                test_f1_list = []
                TPR_list = []
                TNR_list = []
                PPV_list = []

                for repet_num in range(FOLD_NUM):
                    test_acc_list.append(test_result_list[repet_num][1])
                    test_auc_list.append(test_result_list[repet_num][2])
                    test_f1_list.append(test_result_list[repet_num][4])  # [3] mean '1' in F1
                    TPR_list.append(test_result_list[repet_num][5])
                    TNR_list.append(test_result_list[repet_num][6])
                    PPV_list.append(test_result_list[repet_num][7])

                # mean
                test_acc_mean = np.mean(test_acc_list)
                test_auc_mean = np.mean(test_auc_list)
                test_f1_mean = np.mean(test_f1_list)
                test_TPR_mean = np.mean(TPR_list)  # Sensitivity
                test_TNR_mean = np.mean(TNR_list)  # Specificity
                test_PPV_mean = np.mean(PPV_list)  # Precision

                # std
                test_acc_std = np.std(test_acc_list)
                test_auc_std = np.std(test_auc_list)
                test_f1_std = np.std(test_f1_list)
                test_TPR_std = np.std(TPR_list)  # Sensitivity
                test_TNR_std = np.std(TNR_list)  # Specificity
                test_PPV_std = np.std(PPV_list)  # Precision

                with open(SAVE_PATH + 's0epochid_' + str(s0epochid) + '_s1saveepoch_' + str(s1saveepoch) + '_avg3br_a5_mean_test_acc_auc.txt', 'a') as f:
                    f.write(
                        'epoch {}, test_ACC {}, test_AUC {}, test_F1 {}, test_TPR {}, test_TNR {}, test_PPV {}\n'.format(
                            (line_num - 1),
                            (str(format(100 * test_acc_mean, '.2f')) + '±' + str(format(100 * test_acc_std, '.2f'))),
                            (str(format(100 * test_auc_mean, '.2f')) + '±' + str(format(100 * test_auc_std, '.2f'))),
                            (str(format(100 * test_f1_mean, '.2f')) + '±' + str(format(100 * test_f1_std, '.2f'))),
                            (str(format(100 * test_TPR_mean, '.2f')) + '±' + str(format(100 * test_TPR_std, '.2f'))),
                            (str(format(100 * test_TNR_mean, '.2f')) + '±' + str(format(100 * test_TNR_std, '.2f'))),
                            (str(format(100 * test_PPV_mean, '.2f')) + '±' + str(format(100 * test_PPV_std, '.2f')))))
