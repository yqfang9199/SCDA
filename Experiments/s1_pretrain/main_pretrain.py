import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings

warnings.filterwarnings('ignore')

import math
from dataset_prep import DatasetMDD_SRC, DatasetMDD_TGT
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import confusion_matrix
import random
from sklearn.metrics import roc_auc_score
from model import *
import util
import os
import linecache
import re
import glob


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


def train(epoch, model, learning_rate, source_loader, drop, epochs_drop):
    """
    :param epoch: current training epoch
    :param model: defined pre_ddcnet
    :param learning_rate: initial learning rate
    :param source_loader: source loader
    :return:
    """
    log_interval = 1
    LEARNING_RATE = step_decay(epoch, learning_rate, drop, epochs_drop)
    print(f'Learning Rate: {LEARNING_RATE}')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    clf_criterion = nn.CrossEntropyLoss()

    model.train()

    train_correct = 0
    train_loss_accumulate = 0.0

    TN = 0
    FP = 0
    FN = 0
    TP = 0

    tr_auc_y_gt = []
    tr_auc_y_pred = []

    len_dataloader = len(source_loader)
    for step, source_sample_batch in enumerate(source_loader):
        dyn_a, sampling_points = util.bold.process_dynamic_fc(source_sample_batch['timeseries'], window_size=40, window_stride=30, dynamic_length=176)
        sampling_endpoints = [p + 40 for p in sampling_points]

        # generation of "dyn_v" is from original github code
        if step == 0:
            dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=BATCH_SIZE)  # dyn_v (bs, segment_num, 116, 116)
        if not dyn_v.shape[1] == dyn_a.shape[1]:
            dyn_v = repeat(torch.eye(116), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=BATCH_SIZE)
        if len(dyn_a) < BATCH_SIZE:
            dyn_v = dyn_v[:len(dyn_a)]

        t = source_sample_batch['timeseries'].permute(1, 0, 2)  # t.shape [timepoints, batch_size, 116]
        label = source_sample_batch['label']

        logit, _, latent = model(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)  # dyn_v, dyn_a: (batch_size, segment_num, 116, 116)
        loss = clf_criterion(logit, label.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_pred = logit.argmax(1)
        train_prob = logit.softmax(1)[:, 1]  # prob
        train_loss_accumulate += loss

        train_correct += sum(train_pred.data.cpu().numpy() == label.cpu().numpy())

        TN_tmp, FP_tmp, FN_tmp, TP_tmp = confusion_matrix(label.cpu().numpy(), train_pred.data.cpu().numpy(), labels=[0, 1]).ravel()
        TN += TN_tmp
        FP += FP_tmp
        FN += FN_tmp
        TP += TP_tmp
        tr_auc_y_gt.extend(label.cpu().numpy())
        tr_auc_y_pred.extend(train_prob.detach().cpu().numpy())

        if (step + 1) % log_interval == 0:
            print("Train Epoch [{:4d}/{}] Step [{:2d}/{}]: src_cls_loss={:.6f}".format(epoch, TRAIN_EPOCHS, step + 1, len_dataloader, loss.data))

    train_loss_accumulate /= len_dataloader
    train_acc = (TP + TN) / (TP + FP + TN + FN)  # accuracy of each class
    train_AUC = roc_auc_score(tr_auc_y_gt, tr_auc_y_pred)
    train_F1 = (2 * TP) / (2 * TP + FP + FN)

    print('Train set: Average classification loss: {:.4f}, Accuracy: {}/{} ({:.4f}), train_AUC: {:.5}, train_F1: {:.5}'.format(
            train_loss_accumulate, train_correct, (len_dataloader * BATCH_SIZE), train_acc, train_AUC, train_F1))

    # save checkpoint.pth, save train loss and acc to a txt file
    if epoch == 50:
    # if epoch == 50 or epoch == 100 or epoch == 150 or epoch == 200:
        torch.save(model.state_dict(), SAVE_PATH + module + '_' + str(epoch_id) + '_fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth')
    with open(SAVE_PATH + module + '_' + str(epoch_id) + '_fold_' + str(fold) + '_train_loss_and_acc.txt', 'a') as f:
        f.write('epoch {}, classification loss {:.5}, train acc {:.5}, train_AUC {:.5}, train_F1: {:.5}\n'.format(epoch, train_loss_accumulate, train_acc, train_AUC, train_F1))


def test(model, target_loader) -> object:
    """
    :param model: trained alexnet on source data set
    :param target_loader: target dataloader
    :return: correct num
    """

    with torch.no_grad():
        model.eval()
        test_correct = 0
        TN = 0
        FP = 0
        FN = 0
        TP = 0

        te_auc_y_gt = []
        te_auc_y_pred = []

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

            logit, _, _ = model(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)  # dyn_v, dyn_a: (batch_size, segment_num, 116, 116)

            test_pred = logit.argmax(1)
            test_prob = logit.softmax(1)[:, 1]  # prob

            test_correct += sum(test_pred.data.cpu().numpy() == label.cpu().numpy())
            TN_tmp, FP_tmp, FN_tmp, TP_tmp = confusion_matrix(label.cpu().numpy(), test_pred.data.cpu().numpy(), labels=[0, 1]).ravel()
            TN += TN_tmp
            FP += FP_tmp
            FN += FN_tmp
            TP += TP_tmp
            te_auc_y_gt.extend(label.cpu().numpy())
            te_auc_y_pred.extend(test_prob.detach().cpu().numpy())

        TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate
        TNR = TN / (TN + FP)  # Specificity/ true negative rate
        PPV = TP / (TP + FP)  # Precision/ positive predictive value
        test_acc = (TP + TN) / (TP + FP + TN + FN)  # accuracy of each class
        test_AUC = roc_auc_score(te_auc_y_gt, te_auc_y_pred)
        test_F1 = (2 * TP) / (2 * TP + FP + FN)

        print('Test set: Correct_num: {}, test_acc: {:.4f}, test_AUC: {:.4f}, test_F1: {:.4f}, TPR: {:.4f}, TNR: {:.4f}, PPV:{:.4f}\n'.format(
                test_correct, test_acc, test_AUC, test_F1, TPR, TNR, PPV))

        # save test loss and acc to a txt file
        with open(SAVE_PATH + module + '_' + str(epoch_id) + '_fold_' + str(fold) + '_test_loss_and_acc.txt', 'a') as f:
            f.write('epoch {}, test_acc {:.5}, test_AUC {:.5}, test_F1: {:.4f}, TPR {:.5}, TNR {:.5}, PPV {:.5}\n'.format(epoch, test_acc, test_AUC, test_F1, TPR, TNR, PPV))


if __name__ == '__main__':

    ROOT_PATH = '../Data/'
    SAVE_PATH = '../Experiments/s1_pretrain/checkpoints_pretrain/'

    BATCH_SIZE = 175
    TRAIN_EPOCHS = 50
    learning_rate = 0.001
    drop = 0.5
    epochs_drop = 50.0
    FOLD_NUM = 5

    for module in ['mavg']:
        for epoch_id in [5]:
            for fold in range(FOLD_NUM):
                print("module:", module, "  epoch_id", epoch_id, "  fold:", fold)

                seed = fold  # fold
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)  # cpu
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致

                source_dataset = DatasetMDD_SRC(ROOT_PATH)
                source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

                target_dataset = DatasetMDD_TGT(ROOT_PATH)
                target_test_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

                print('Construct model begin:')
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # print('device:', device)
                pre_ddcnet = ModelSTAGIN(input_dim=116, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.5, cls_token='sum', readout='sero')
                pre_ddcnet.load_state_dict(torch.load('../Experiments/s0_unsup_tr/checkpoint_unsup_tr/unsuptr_' + module + '_fold_0_epoch_' + str(epoch_id) + '.pth', map_location=device))

                pre_ddcnet.to(device)

                with open(SAVE_PATH + module + '_' + str(epoch_id) + '_fold_' + str(fold) + '_train_loss_and_acc.txt', 'a') as f:
                    f.write('total_epoch: {}, batch_size: {}, initial_lr {:.8}, drop_lr: {:.5}, drop_lr_per_epoch\n'.format(TRAIN_EPOCHS, BATCH_SIZE, learning_rate, drop, epochs_drop))
                with open(SAVE_PATH + module + '_' + str(epoch_id) + '_fold_' + str(fold) + '_test_loss_and_acc.txt', 'a') as f:
                    f.write('total_epoch: {}, batch_size: {}, initial_lr {:.8}, drop_lr: {:.5}, drop_lr_per_epoch\n'.format(TRAIN_EPOCHS, BATCH_SIZE, learning_rate, drop, epochs_drop))

                for epoch in range(1, TRAIN_EPOCHS + 1):
                    print(f'Train Epoch {epoch}:')
                    train(epoch, pre_ddcnet, learning_rate, source_loader, drop, epochs_drop)
                    correct = test(pre_ddcnet, target_test_loader)

            # calculate the mean value of five folds
            def get_line_context(file_path, line_number=(TRAIN_EPOCHS + 1)):
                return linecache.getline(file_path, line_number).strip()

            # a5
            for line_num in range(2, TRAIN_EPOCHS + 2):
                test_result_list = []
                for fold in range(FOLD_NUM):
                    txt_file_path = SAVE_PATH + module + '_' + str(epoch_id) + '_fold_' + str(fold) + '_test_loss_and_acc.txt'
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

                with open(SAVE_PATH + module + '_' + str(epoch_id) + '_a5_mean_test_acc_auc.txt', 'a') as f:
                    f.write(
                        'epoch {}, test_ACC {}, test_AUC {}, test_F1 {}, test_TPR {}, test_TNR {}, test_PPV {}\n'.format(
                            (line_num - 1),
                            (str(format(100 * test_acc_mean, '.2f')) + '±' + str(format(100 * test_acc_std, '.2f'))),
                            (str(format(100 * test_auc_mean, '.2f')) + '±' + str(format(100 * test_auc_std, '.2f'))),
                            (str(format(100 * test_f1_mean, '.2f')) + '±' + str(format(100 * test_f1_std, '.2f'))),
                            (str(format(100 * test_TPR_mean, '.2f')) + '±' + str(format(100 * test_TPR_std, '.2f'))),
                            (str(format(100 * test_TNR_mean, '.2f')) + '±' + str(format(100 * test_TNR_std, '.2f'))),
                            (str(format(100 * test_PPV_mean, '.2f')) + '±' + str(format(100 * test_PPV_std, '.2f')))))
