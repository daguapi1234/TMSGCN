
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import normalize_A, generate_A
from utils import cheb_conv_v3
from utils import get_distance_adj


def FFT_for_Period(x, k=5):
    N = x.size()[1]
    fft_values = torch.fft.rfft(x, dim=1)
    fft_freq = torch.fft.rfftfreq(N, d=1.0/120)
    fft_mag = torch.abs(fft_values).mean(-1)
    fft_mag[:, 0] = 0.
    _, top_fmag = torch.topk(fft_mag, k)
    top_freq = fft_freq[top_fmag]
    top_scale = x.shape[1] // top_freq
    top_scale = top_scale.detach().cpu().numpy().astype(int)

    f = abs(fft_values).mean(-1)
    f_top = f[torch.arange(f.size(0)).unsqueeze(1), top_fmag.detach().cpu().numpy()]
    return top_scale, f_top


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class MIC(nn.Module):
    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.src_mask = None
        self.conv_kernel = conv_kernel
        self.isometric_kernel = isometric_kernel
        self.device = device

        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=78//i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        # self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))

        self.fnn = FeedForwardNetwork(feature_size, feature_size * 4, dropout)
        self.fnn_norm = torch.nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, channel, seq_len = input.shape
        x = input
        x1 = self.drop(self.act(conv1d(x)))
        x = x1
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=self.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        if x.shape != x1.shape:
            x = x[:, :, 0:x1.shape[2]]
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]
        x = self.norm((x + input).permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def forward(self, src):
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out = src[:, :, 0, :]
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)
        mg = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(2)), dim=2)
        mg = self.merge(mg).squeeze(-2).permute(0, 2, 1)
        mgn = self.fnn_norm(mg + self.fnn(mg)).permute(0, 2, 1).unsqueeze(2)

        return mgn


class Seasonal_Prediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(Seasonal_Prediction, self).__init__()

        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                                   decomp_kernel=decomp_kernel,conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, device=device)
                                      for i in range(d_layers)])

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)

        return dec


class AGCN_Block(nn.Module):
    def __init__(self, xdim, k_adj, GCN_num_out, dis_adj):
        super(AGCN_Block, self).__init__()
        self.sdim = xdim[1]
        self.tdim = xdim[-1]
        self.LN_s1 = nn.LayerNorm(normalized_shape=[self.sdim, self.tdim])  # for空间掩码图
        self.BN_s = nn.BatchNorm1d(self.sdim)

        self.K = k_adj
        self.GCN = cheb_conv_v3(k_adj, GCN_num_out, GCN_num_out)

        self.distance_adj = dis_adj

        self.s_li = nn.Linear(self.tdim, self.tdim)
        self.n_max_s = self.sdim // 5

    def forward(self, x):
        x_s = self.LN_s1(x)
        A_s = generate_A(self.s_li(x_s), self.sdim, self.n_max_s)
        x_s = self.BN_s(x_s)
        L_DS = A_s + self.distance_adj
        L_s = normalize_A(L_DS)
        result_s = self.GCN(x_s, L_s)

        return result_s


class AGCN(nn.Module):
    def __init__(self, xdim, k_adj, fft_k, cin, cout, GCN_num_out, dis_adj):
        super(AGCN, self).__init__()
        self.sdim = xdim[1]
        self.tdim = xdim[-1]
        self.cdim = 1
        self.distance_adj = dis_adj
        self.gcn_list = nn.ModuleList([AGCN_Block(xdim, k_adj, GCN_num_out, dis_adj) for _ in range(self.cdim)])

    def forward(self, x):
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        output = []
        for i in range(in_channels):
            x_i = x[:, :, i, :]
            x_i_out = self.gcn_list[i](x_i)
            output.append(x_i_out)
        output = torch.stack(output, dim=2)

        return output


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[0:1, :].repeat((self.kernel_size - 1) // 2, 1)
        end = x[-1:, :].repeat((self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=0)
        x_pad = torch.unsqueeze(x_pad, dim=0)
        x_pad = self.avg(x_pad.permute(0, 2, 1))
        x_pad = x_pad.permute(0, 2, 1)
        x_pad = torch.squeeze(x_pad, dim=0)
        if x.shape[0] != x_pad.shape[0]:
            if x.shape[0] > x_pad.shape[0]:
                pad = x_pad[-1:, :].repeat(x.shape[0] - x_pad.shape[0], 1)
                x_pad = torch.cat([x_pad, pad], dim=0)
            else:
                x_pad = x_pad[0:x.shape[0], :]

        return x_pad


class FFT_TREND(nn.Module):
    def __init__(self, xdim, K):
        super(FFT_TREND, self).__init__()

        self.time_length = xdim[2]
        self.node_length = xdim[1]
        self.trd_nb = K
        self.out_all = []

    def forward(self, x):
        x = x.permute(0, 2, 1)
        period, f = FFT_for_Period(x, k=self.trd_nb)
        out_all = []
        for i in range(0, period.shape[1]):
            scale = period[:, i]
            out = []
            out_res = []
            for j in range(x.shape[0]):
                kernel_size = scale[j]
                ma_optor = moving_avg(kernel_size, stride=1)
                moving_mean = ma_optor(x[j])
                res = x[j] - moving_mean
                out.append(moving_mean)
                out_res.append(res)
            out = torch.stack(out, dim=0)
            out_res = torch.stack(out_res, dim=0)

            out_all.append(out)
        self.out_all = torch.stack(out_all, dim=0)
        self.out_all = self.out_all.permute(1, 3, 0, 2)

        return self.out_all


class trend_stgcn(nn.Module):

    def __init__(self, X, k_adj, GCN_numout, num_class, trd_k, cin, cout, ker_set, tm_len, blocks=2, layers=1, dilation_factor=2):
        super(trend_stgcn, self).__init__()

        xdim = X.shape
        self.sdim = xdim[1]
        self.tdim = xdim[-1]
        self.cdim = xdim[2]
        self.device = X.device

        elc_loc_path = '/data/wangjincen/P300_TST/bci3/eloc64.txt'
        self.distance_adj = get_distance_adj(elc_loc_path, circumference=0.56, a=0.2)
        self.distance_adj = torch.from_numpy(self.distance_adj).to(self.device)
        self.distance_adj = self.distance_adj.to(torch.float32)

        self.blocks = blocks
        self.trend_gen = FFT_TREND(xdim, trd_k)
        self.tnet = nn.ModuleList(
            [Seasonal_Prediction(embedding_size=self.sdim, conv_kernel=ker_set, isometric_kernel=ker_set, device=self.device) for _ in range(self.blocks)])
        self.agcn = nn.ModuleList([AGCN(xdim, k_adj, trd_k, cin, cout, GCN_numout, self.distance_adj) for _ in range(self.blocks)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(1) for _ in range(self.blocks)])

        self.final_conv = nn.Conv2d(in_channels=cout, out_channels=1, kernel_size=(1, 1), bias=True)
        self.fc1 = nn.Linear(self.sdim*GCN_numout, 64)
        self.fc2 = nn.Linear(64, num_class)
        self.drop_conv = nn.Dropout(0.3)
        self.dropfc = nn.Dropout(0.1)
        self.BN_class = nn.BatchNorm1d(num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        bs, num_nodes, samples = x.shape

        trends = self.trend_gen(x)
        x = torch.unsqueeze(x, dim=2)
        x_wt = torch.cat((x, trends), dim=2)
        x_wt = self.final_conv(x_wt.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        for i in range(self.blocks):
            x_res = x_wt
            x_wt = self.tnet[i](x_wt)
            x_wt = self.agcn[i](x_wt)

            x_wt = x_res + x_wt

            x_wt = x_wt.permute(0, 2, 1, 3)
            x_wt = self.bn[i](x_wt)
            x_wt = x_wt.permute(0, 2, 1, 3)

        result = x_wt.permute(0, 2, 1, 3)
        result = result.reshape(bs, -1)
        result = self.dropfc(F.relu(self.fc1(result)))
        result = self.fc2(result)
        result = self.BN_class(result)
        result_softmax = self.softmax(result)

        return result_softmax, result


if __name__ == '__main__':
    X = torch.rand(32, 64, 78).cuda()
    ker_set = [3, 5, 7]  # 235  2367 Connecting那篇推荐
    trd_k = 1
    cin = cout = trd_k + 1
    net = trend_stgcn(X, k_adj=3, GCN_numout=78, num_class=2, trd_k=trd_k, cin=cin, cout=cout, ker_set=ker_set,
                      tm_len=78, blocks=1, layers=1, dilation_factor=2)
    net = net.cuda()
    print(net)
    Y, _ = net(X)
    print(Y.size())

    loaded_adj = np.load('/data/wangjincen/P300_TST/bci3/adjacency_matrix_bci3.npy')
    print(loaded_adj)