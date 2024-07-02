import os
import csv
import numpy as np
from sklearn import metrics


class LoggerSTAGIN(object):
    def __init__(self, k_fold=None, num_classes=None):
        super().__init__()
        self.k_fold = k_fold
        self.num_classes = num_classes
        self.initialize(k=None)

    def __call__(self, **kwargs):
        if len(kwargs) == 0:
            self.get()
        else:
            self.add(**kwargs)

    def _initialize_metric_dict(self):
        return {'pred': [], 'true': [], 'prob': []}

    def initialize(self, k=None):
        if self.k_fold is None and k is None:
            self.samples = {}
            for _k in range(20):  # should be same as repeat_num
                self.samples[_k] = self._initialize_metric_dict()
        else:
             self.samples[k] = self._initialize_metric_dict()

        # if self.k_fold is None:
        #     self.samples = self._initialize_metric_dict()
        # else:
        #     if k is None:
        #         self.samples = {}
        #         for _k in range(self.k_fold):
        #             self.samples[_k] = self._initialize_metric_dict()
        #     else:
        #         self.samples[k] = self._initialize_metric_dict()

    def add(self, k=None, **kwargs):
        for sample, value in kwargs.items():
            self.samples[k][sample].append(value)
        # if self.k_fold is None:
        #     for sample, value in kwargs.items():
        #         self.samples[sample].append(value)
        # else:
        #     assert k in list(range(self.k_fold))
        #     for sample, value in kwargs.items():
        #         self.samples[k][sample].append(value)

    def get(self, k=None, initialize=False):

        if k is None:
            true, pred, prob = {}, {}, {}
            # for k in range(self.k_fold):
            for k in range(6):
                true[k] = np.concatenate(self.samples[k]['true'])
                pred[k] = np.concatenate(self.samples[k]['pred'])
                prob[k] = np.concatenate(self.samples[k]['prob'])
        else:
            true = np.concatenate(self.samples[k]['true'])
            pred = np.concatenate(self.samples[k]['pred'])
            prob = np.concatenate(self.samples[k]['prob'])

        # if self.k_fold is None:
        #     true = np.concatenate(self.samples['true'])
        #     pred = np.concatenate(self.samples['pred'])
        #     prob = np.concatenate(self.samples['prob'])
        # else:
        #     if k is None:
        #         true, pred, prob = {}, {}, {}
        #         for k in range(self.k_fold):
        #             true[k] = np.concatenate(self.samples[k]['true'])
        #             pred[k] = np.concatenate(self.samples[k]['pred'])
        #             prob[k] = np.concatenate(self.samples[k]['prob'])
        #     else:
        #         true = np.concatenate(self.samples[k]['true'])
        #         pred = np.concatenate(self.samples[k]['pred'])
        #         prob = np.concatenate(self.samples[k]['prob'])

        if initialize:
            self.initialize(k)

        return dict(true=true, pred=pred, prob=prob)

    def evaluate(self, k=None, initialize=False, option='mean'):
        samples = self.get(k)

        # if not self.k_fold is None and k is None:

        if option == 'mean':
            aggregate = np.mean
        elif option == 'std':
            aggregate = np.std
        else:
            raise

        if k is None:
            # accuracy = aggregate([metrics.accuracy_score(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
            # precision = aggregate([metrics.precision_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes == 2 else 'micro') for k in range(self.k_fold)])
            # recall = aggregate([metrics.recall_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes == 2 else 'micro') for k in range(self.k_fold)])
            #
            # # print(np.isinf(samples['prob'][k][:,1]).all())
            # roc_auc = aggregate([metrics.roc_auc_score(samples['true'][k], samples['prob'][k][:, 1]) for k in range(self.k_fold)]) if self.num_classes == 2 else np.mean(
            #     [metrics.roc_auc_score(samples['true'][k], samples['prob'][k], average='macro', multi_class='ovr') for k in range(self.k_fold)])
            # f1_score = aggregate([metrics.f1_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes == 2 else 'micro') for k in range(self.k_fold)])

            accuracy = aggregate([metrics.accuracy_score(samples['true'][k], samples['pred'][k]) for k in range(6)])
            precision = aggregate([metrics.precision_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes == 2 else 'micro') for k in range(6)])
            recall = aggregate([metrics.recall_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes == 2 else 'micro') for k in range(6)])

            # print(np.isinf(samples['prob'][k][:,1]).all())
            roc_auc = aggregate([metrics.roc_auc_score(samples['true'][k], samples['prob'][k][:, 1]) for k in range(6)]) if self.num_classes == 2 else np.mean(
                [metrics.roc_auc_score(samples['true'][k], samples['prob'][k], average='macro', multi_class='ovr') for k in range(6)])
            f1_score = aggregate([metrics.f1_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes == 2 else 'micro') for k in range(6)])
        else:
            accuracy = metrics.accuracy_score(samples['true'], samples['pred'])
            precision = metrics.precision_score(samples['true'], samples['pred'], average='binary' if self.num_classes == 2 else 'micro')
            # print(samples['prob'][:,1].shape)
            recall = metrics.recall_score(samples['true'], samples['pred'], average='binary' if self.num_classes == 2 else 'micro')
            roc_auc = metrics.roc_auc_score(samples['true'], samples['prob'][:, 1]) if self.num_classes == 2 else metrics.roc_auc_score( samples['true'], samples['prob'], average='macro', multi_class='ovr')
            f1_score = metrics.f1_score(samples['true'], samples['pred'], average='binary' if self.num_classes == 2 else 'micro')


        # else:
        #     accuracy = metrics.accuracy_score(samples['true'], samples['pred'])
        #     precision = metrics.precision_score(samples['true'], samples['pred'],
        #                                         average='binary' if self.num_classes == 2 else 'micro')
        #     # print(samples['prob'][:,1].shape)
        #     # print(samples['prob'][:,1])
        #     recall = metrics.recall_score(samples['true'], samples['pred'],
        #                                   average='binary' if self.num_classes == 2 else 'micro')
        #     roc_auc = metrics.roc_auc_score(samples['true'],
        #                                     samples['prob'][:, 1]) if self.num_classes == 2 else metrics.roc_auc_score(
        #         samples['true'], samples['prob'], average='macro', multi_class='ovr')
        #     f1_score = metrics.f1_score(samples['true'], samples['pred'],
        #                                 average='binary' if self.num_classes == 2 else 'micro')

        if initialize:
            self.initialize(k)

        return dict(accuracy=accuracy, precision=precision, recall=recall, roc_auc=roc_auc, f1_score=f1_score)

    def to_csv(self, targetdir, k=None, initialize=False):
        metric_dict = self.evaluate(k, initialize)  # 平均所有折的性能度量
        append = os.path.isfile(os.path.join(targetdir, 'metric.csv'))
        with open(os.path.join(targetdir, 'metric.csv'), 'a', newline='') as f:
            writer = csv.writer(f)  # 写入csv文件
            if not append:
                writer.writerow(['fold'] + [str(key) for key in metric_dict.keys()])  # writerow()将一个列表全部写入csv的同一行。
            writer.writerow([str(k)] + [str(value) for value in metric_dict.values()])
            if k is None:
                writer.writerow([str(k)] + list(self.evaluate(k, initialize, 'std').values()))
