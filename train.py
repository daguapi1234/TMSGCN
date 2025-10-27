import argparse  # 命令行解析模块
import datetime
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from data_loader import load_dataloader
from utils import evaluate
import copy
import time
from scipy import io as scio

from model import trend_stgcn

torch.cuda.set_device(1)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train a network for EEG_P300.")

    parser.add_argument(
        '-lr',
        '--lr',
        type=float,
        default=1e-3,
        help='Spcifies learing rate for optimizer. (default: 1e-3)')
    parser.add_argument(
        '-ep',
        '--epochs',
        type=int,
        default=1,
        help='Number of training epochs. (default: 50)'
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for data loaders. (default: 64)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers for data loader. (default: 8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=4567,
        help='Number of workers for data loader. (default: 8)'
    )
    parser.add_argument(
        '--type',
        type=int,
        default=0,
        help='Number of workers for data loader. (default: 8)'
    )
    parser.add_argument(
        '--blocks',
        type=int,
        default=1,
        help='blocks'
    )
    parser.add_argument(
        '--layers',
        type=int,
        default=1,
        help='layers'
    )
    parser.add_argument(
        '--fft_k',
        type=int,
        default=3,
        help='fft_k'
    )
    opt = parser.parse_args()

    seed = opt.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # 是否混淆矩阵 及 混淆矩阵结果图名称
    draw_CM = 'False'    # Ture or False
    picture_name = 'shuffle_82_CM'

    xdim = [opt.batch_size, 64, 78]
    k_adj = 3
    GCN_num_out = xdim[2]

    acc_mean = 0
    acc_all = []

    loaded_data_A = np.load('./Subject_A_signal_train.npy')
    loaded_label_A = np.load('./Subject_A_type_train.npy')
    x_load = loaded_data_A
    y_load = loaded_label_A
    x_load = torch.from_numpy(x_load)
    y_load = torch.from_numpy(y_load)
    print('x_load', x_load.shape)
    print('y_load', y_load.shape)
    x_load = x_load.reshape(-1, x_load.shape[2], x_load.shape[3])
    x_load = x_load.permute((0, 2, 1))
    y_load = y_load.reshape(y_load.shape[0]*y_load.shape[1])
    print('x_load', x_load.shape)
    print('y_load', y_load.shape)

    loaded_data_A = np.load('./Subject_A_signal_test.npy')
    loaded_label_A = np.load('./Subject_A_type_test.npy')
    x_load_test = loaded_data_A
    y_load_test = loaded_label_A
    x_load_test = torch.from_numpy(x_load_test)
    y_load_test = torch.from_numpy(y_load_test)
    print('x_load_test', x_load_test.shape)
    print('y_load_test', y_load_test.shape)
    x_load_test = x_load_test.reshape(-1, x_load_test.shape[2], x_load_test.shape[3])
    x_load_test = x_load_test.permute((0, 2, 1))
    y_load_test = y_load_test.reshape(y_load_test.shape[0] * y_load_test.shape[1])
    print('x_load_test', x_load_test.shape)
    print('y_load_test', y_load_test.shape)

    train_data, test_data, train_label, test_label = x_load, x_load_test, y_load, y_load_test
    print('train_data', train_data.shape)
    print('train_label', train_label.shape)
    print('test_data', test_data.shape)
    print('test_label', test_label.shape)
    train_data, test_data, train_label, test_label = train_data.cuda(), test_data.cuda(), train_label.cuda(), test_label.cuda()

    model_selected = trend_stgcn

    ker_set = [8, 15, 30, 60]

    fft_k = opt.fft_k
    model = model_selected(X=train_data, k_adj=3, GCN_numout=GCN_num_out, num_class=2, trd_k=fft_k, cin=fft_k+1, cout=fft_k+1, ker_set=ker_set,
                      tm_len=train_data.shape[-1], blocks=opt.blocks, layers=opt.layers, dilation_factor=2).cuda()
    model_name = model.__class__.__name__
    print(model)

    train_iter, test_iter = load_dataloader(train_data, test_data, train_label, test_label, opt)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),
                        lr=opt.lr
    )

    # ---------- Train --------------
    print('began training on', device, '.........')
    start_time_all = time.time()

    acc_test_best = 0.0
    f1_test_best = 0.0
    ep_best = 0
    n = 0
    cmx_best = []

    for ep in range(opt.epochs):
        print('Epoch {} beginning······'.format(ep+1))

        # Train...
        model.train()

        pbar = tqdm(enumerate(train_iter),
                    total=len(train_iter), ncols=100)

        n += 1
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.0

        for i, data in pbar:
            images, labels = data

            images = images.float()
            labels = labels

            output, fc_feature = model(images)

            loss = criterion(fc_feature, labels.long())

            pred = output.argmax(dim=1)

            correct += (pred == labels).sum().item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss

            # 优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute computation time and *compute_efficiency*
            pbar.set_description(
                f'Acc: {accuracy:.4f}, '
                f'loss: {loss.item():.2f}, epoch:{ep+1}/{opt.epochs}')
            start_time = time.time()

        # Evaluate...
        acc_test, f1_test, cmx = evaluate(test_iter, model, acc_test_best, f1_test_best,
                                                                   picture_name, draw_CM)

        if f1_test >= f1_test_best:
            f1_test_best = f1_test
            acc_test_best = acc_test
            model_best = model
            best_model_weight = copy.deepcopy(model.state_dict())
            cmx_best = cmx
            ep_best = ep + 1
            n = 0

        if n >= opt.epochs // 10 and f1_test < f1_test_best - 0.1:
            print('----------------------reload------------------------')
            n = 0
            model = model_best

        print('>>> best test Accuracy: {}'.format(acc_test_best))
        print('>>> best test F1: {}'.format(f1_test_best))
        print('>>> best cmx: {}'.format(cmx_best))
        print('>>> best epoch of acc: {}'.format(ep_best))
        print('----------------------------------------------')

    print("Training end")

    result_path = './result/'
    file_name = ('{}-ep{}-bs{}-'.format(model_name,opt.epochs,opt.batch_size)
                 + "{:.2f}".format(f1_test_best) + '.mat')

    scio.savemat(result_path + file_name,
                 {'epochs': opt.epochs,
                  'batch_size': opt.batch_size,
                  'acc_best': acc_test_best,
                  'f1_best': f1_test_best,
                  'best_epoch': ep_best,
                  'best_cmx': cmx_best,
                  })

