import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat


class ModuleTimestamping(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        # The input of nn.GRU is (timepoints, batchsize, 116). For each timepoint, the input feature is 1-D vector (116)

    def forward(self, t, sampling_endpoints):
        # t.shape: (timepoints, batch_size, 116)
        # sampling_endpoints: [40, 43, 46, 49 ..., timepoints]
        return self.rnn(t[:sampling_endpoints[-1]])[0][[p - 1 for p in sampling_endpoints]]  # (segment_num, batch_size, 64)


class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon:
            self.epsilon = nn.Parameter(torch.Tensor([[0.0]]))  # assumes that the adjacency matrix includes self-loop
        else:
            self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())

    def forward(self, v, a):
        # v.shape: (batch_size * segment_num * 116, 64)
        # a.shape: (batch_size * segment_num * 116, batch_size * segment_num * 116)
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v  # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine


class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1, 1, 1], dtype=torch.float32)


class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale * hidden_dim)),
                                   nn.BatchNorm1d(round(upscale * hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale * hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):  # x.shape: (segment_num, batch_size, 116, 64)
        x_readout = x.mean(node_axis)  # x_readout.shape: (segment_num, batch_size, 64)
        x_shape = x_readout.shape  # x_shape: (segment_num, batch_size, 64)
        x_embed = self.embed(x_readout.reshape(-1, x_shape[-1]))  # x_embed.shape: (segment_num * batch_size, 64)
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1], -1)  # x_graphattention.shape: (segment_num, batch_size, 116)
        permute_idx = list(range(node_axis)) + [len(x_graphattention.shape) - 1] + list(range(node_axis, len(x_graphattention.shape) - 1))  # permute_idx: [0, 1, 2]
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)


class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(
            torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n')) / np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)


class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, input_dim))

    def forward(self, x):  # x.shape: (segment_num, batch_size, 64)
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend)  # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix
        # x_attend.shape: (segment_num, batch_size, 64), attn_matrix.shape: (batch_size, segment_num, segment_num)


class ModelSTAGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2, sparsity=30, dropout=0.5,
                 cls_token='sum', readout='sero'):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token == 'sum':
            self.cls_token = lambda x: x.sum(0)
        elif cls_token == 'mean':
            self.cls_token = lambda x: x.mean(0)
        elif cls_token == 'param':
            self.cls_token = lambda x: x[-1]
        else:
            raise
        if readout == 'garo':
            readout_module = ModuleGARO
        elif readout == 'sero':
            readout_module = ModuleSERO
        elif readout == 'mean':
            readout_module = ModuleMeanReadout
        else:
            raise

        self.num_classes = num_classes
        self.sparsity = sparsity

        # define modules
        self.timestamp_encoder = ModuleTimestamping(input_dim, hidden_dim, hidden_dim)
        self.initial_linear = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList()  # 它可以以列表的形式来存储多个子模块
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))
            self.transformer_modules.append(ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

    def _collate_adjacency(self, a):
        # a.shape: (batch_size, segment_num, 116, 116). It calculates FC matrix for each sliding window
        i_list = []
        v_list = []
        for sample, _dyn_a in enumerate(a):  # _dyn_a.shape: (segment_num, 116, 116)
            for timepoint, _a in enumerate(_dyn_a):  # _a.shape: (116, 116)
                thresholded_a = (_a > np.percentile(_a.detach().cpu().numpy(), 100 - self.sparsity))  # 稀疏, 不大于70的为0
                _i = thresholded_a.nonzero(as_tuple=False)  # 非零元素坐标
                # _i.shape: (4036, 2), here 4036 is the number of top 30% elements, 2 corresponds to x_axis and y_axis

                _v = torch.ones(len(_i))  # _v.shape:  (4036,) and all elements are 1
                _i += sample*a.shape[1]*a.shape[2] + timepoint*a.shape[2]  # batchID*54*116 + segment_numID*116
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        return torch.sparse.FloatTensor(_i, _v, (a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3]))  # (batch_size * segment_num * 116, batch_size * segment_num * 116)

    def forward(self, v, a, t, sampling_endpoints):
        # v/a.shape: (batch_size, segment_num, 116, 116)
        # t.shape: (timepoints, batch_size, 116)
        # sampling_endpoints: [40, 43, 46, 49 ..., timepoints]

        logit = 0.0
        attention = {'node-attention': [], 'time-attention': []}
        feat_G_list = []
        feat_T_list = []
        feat_L_list = []
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]

        h1 = v  # v: one-hot encoding (constant), h1: (batch_size, segment_num, 116, 116)
        # h1 = torch.cat([v, time_encoding], dim=3)  # v: one-hot encoding (constant), h1: (batch_size, segment_num, 116, 116+64)
        h2 = rearrange(h1, 'b t n c -> (b t n) c')  # h2.shape: (batch_size * segment_num * 116, 116+?0/64)
        h3 = self.initial_linear(h2)  # 全连接层, h3.shape: (batch_size * segment_num * 116, 64)

        a1 = self._collate_adjacency(a)  # a1.shape: (batch_size * segment_num * 116, batch_size * segment_num * 116)

        for layer, (G, R, T, L) in enumerate(
                zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h = G(h3, a1)  # h.shape: (batch_size * segment_num * 116, 64)
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)  # h_bridge.shape: (segment_num, batch_size, 116, 64)
            h_readout, node_attn = R(h_bridge, node_axis=2)
            # h_readout.shape: (segment_num, batch_size, 64)
            # node_attn.shape: (batch_size, segment_num, 116) if sero or garo; otherwise node_attn=0 (i.e., mean)

            feat_G = h_readout.mean(0)  # feat_G.shape: (batch_size, 64)
            h_attend, time_attn = T(h_readout)  # h_readout.shape: (segment_num, batch_size, 64), h_attend.shape: (segment_num, batch_size, 64), time_attn.shape: (batch_size, segment_num, segment_num)

            feat_T = self.cls_token(h_attend)  # feat_T.shape: (batch_size, 64)
            feat_L = L(feat_T)  # feat_L.shape: (batch_size, 2)
            logit += self.dropout(L(feat_T))  # logit.shape: (batch_size, 2)

            attention['node-attention'].append(node_attn)
            attention['time-attention'].append(time_attn)
            feat_G_list.append(feat_G)
            feat_T_list.append(feat_T)
            feat_L_list.append(feat_L)

        attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()  # 把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()  # dim是选择生成的维度
        feat_G = torch.stack(feat_G_list, dim=1)  # (batch_size, 2, 64)
        feat_T = torch.stack(feat_T_list, dim=1)  # (batch_size, 2, 64)
        feat_L = torch.stack(feat_L_list, dim=1)  # (batch_size, 2, 2)

        return logit, attention, feat_G, feat_T, feat_L
        # logit.shape: (batch_size, 2)
        # attention['node-attention'].shape: (batch_size, 2, segment_num, 116)
        # attention['time-attention'].shape: (batch_size, 2, segment_num, segment_num)
        # latent.shape: (batch_size, 2, 64)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
