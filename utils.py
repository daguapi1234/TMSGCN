import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


class cheb_conv_v3(nn.Module):
    def __init__(self, K, in_channels, out_channels):
        super(cheb_conv_v3, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda()) for _ in range(self.K)])
        for i in range(self.K):
            nn.init.kaiming_normal_(self.Theta[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x, L_tilde):

        batch_size, num_of_vertices, num_of_timesteps = x.shape
        cheb_polynomials = generate_cheby_adj(L_tilde, self.K)
        output = torch.zeros(batch_size, num_of_vertices, self.out_channels).cuda()  #

        for k in range(self.K):
            T_k = cheb_polynomials[k]
            theta_k = self.Theta[k]
            rhs = torch.matmul(T_k, x)
            output = output + torch.matmul(rhs, theta_k)

        output = F.relu(output)

        return output


def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:    # T0
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:  # T1
            support.append(L)
        else:         # Tk
            # temp = torch.matmul(2*L,support[-1],)-support[-2]
            temp = 2 * L * support[-1] - support[-2]
            support.append(temp)
    return support


def generate_A(x, n_vertex, n_max):

    features_a = F.normalize(x, p=2, dim=-1)
    adj = torch.matmul(features_a, features_a.transpose(2, 1))
    adj = torch.softmax(adj, 2)
    mask = torch.zeros(adj.size(0), n_vertex, n_vertex, device=x.device)  # .to(device)
    mask.fill_(0.0)
    s, t = adj.topk(n_max, 2)
    mask.scatter_(2, t, s.fill_(1))
    A = adj * mask

    return A


def normalize_A(A, lmax=2):
    A = F.relu(A)
    N = A.shape[1]
    A = A * (torch.ones(N, N).cuda() - torch.eye(N, N).cuda()) 
    A = A + A.transpose(1,2)

    d = torch.sum(A, 2)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)

    L = torch.eye(N, N).cuda() - torch.matmul(torch.matmul(D, A), D)
    Lnorm = (2 * L / lmax) - torch.eye(N, N).cuda()

    return Lnorm


def get_distance_adj(filepath, circumference=0.56, r_max_original=0.5, a=0.2):
    r_max_real = circumference / 0.6283185
    scale_factor = r_max_real / r_max_original

    data = np.loadtxt(filepath, dtype={'names': ('index', 'degrees', 'radius', 'label'),
                                       'formats': ('i4', 'f4', 'f4', 'S10')})

    degrees = data['degrees']
    radius = data['radius']
    labels = [str(label.decode('utf-8')).strip('.') for label in data['label']]  # 第四列是电极名称
    theta = np.deg2rad(degrees)
    radius_real = radius * scale_factor
    x = radius_real * np.cos(theta)
    y = radius_real * np.sin(theta)
    z = np.sqrt(r_max_real ** 2 - radius_real ** 2)

    distances = []
    pairs = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            distance = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)

            distances.append(distance)
            pairs.append((i, j))

    distances = np.array(distances)
    pairs = np.array(pairs)

    num_pairs = len(distances)
    top_20_percent_count = round(a * num_pairs)
    sorted_indices = np.argsort(distances)

    top_20_percent_distances = distances[sorted_indices[:top_20_percent_count]]
    top_20_percent_pairs = pairs[sorted_indices[:top_20_percent_count]]

    for k in range(top_20_percent_count):
        i, j = top_20_percent_pairs[k]  # 获取电极对的索引
        # print(f'电极对: {i} - {j}, 距离: {top_20_percent_distances[k]:.2f}')
    distance_threshold = top_20_percent_distances[-1]

    adjacency_matrix = np.zeros((len(x), len(x)))
    dis_matrix = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            distance = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)
            dis_matrix[i, j] = distance
    distances = dis_matrix[~np.isinf(dis_matrix)].flatten()
    std = distances.std()
    adjacency_matrix = np.exp(-np.square(dis_matrix / std))
    adjacency_matrix[dis_matrix > 0.9] = 0

    return adjacency_matrix


class DrawConfusionMatrix:
    def __init__(self, labels_name, picture_name, normalize=True):
        """
        normalize：是否设元素为百分比形式
        """
        self.normalize = normalize
        self.labels_name = labels_name
        self.picture_name = picture_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, labels, predicts):
        """
        :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        :return:
        """

        for predict, label in zip(labels, predicts):
            self.matrix[label, predict] += 1

    def getMatrix(self, normalize=True):
        """
        根据传入的normalize判断要进行percent的转换，
        如果normalize为True，则矩阵元素转换为百分比形式，
        如果normalize为False，则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵
        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])  # 百分比转换
            self.matrix = np.around(self.matrix, 2)  # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def drawMatrix(self):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        # bbox_inches='tight' 确保标签信息显示全
        plt.savefig('/data2/zhaotao/EEG_P300/result/' + self.picture_name, bbox_inches='tight')
        plt.show()


def evaluate(test_iter, model, acc_test_best, f1_test_best, picture_name, draw_CM):
    # Eval
    model.eval()  # 测试阶段，不启用模型中的 Batch Normalization 和 Dropout

    labels_name = ['0', '1']
    drawconfusionmatrix = DrawConfusionMatrix(labels_name, picture_name)

    correct, total = 0, 0
    acc_all = []
    true_label_list = []
    pred_label_list = []

    pbar = tqdm(enumerate(test_iter),
                total=len(test_iter),ncols=100)

    with torch.no_grad():
        for i, data in pbar:
            # data preparation
            images, labels = data
            images = images.float().cuda()
            labels = labels.cuda()

            output, fc_feature = model(images)
            pred = output.argmax(dim=1)

            correct += (pred == labels).sum().item()
            acc_all.append((pred == labels).sum().item() / len(labels))

            # 将tensor转为绘制混淆矩阵所需的numpy数据类型
            predict_np = output.cpu()
            predict_np = predict_np.detach().numpy()
            predict_np = np.argmax(predict_np, axis=-1)

            labels_np = labels.cpu()
            labels_np = labels_np.detach().numpy()

            true_label_list.append(labels_np)
            pred_label_list.append(predict_np)

            if draw_CM == 'True':
                drawconfusionmatrix.update(labels_np, predict_np)  # 将新批次的predict和label更新（保存）

            total += len(labels)

    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='binary')[:-1]
    cmx = confusion_matrix(y_true, y_pred)

    print("> accuracy: ", accuracy)
    print("> cmx: ", cmx)
    print("> f1: ", f1)

    acc_test = correct / total

    if draw_CM=='True' and acc_test >= acc_test_best :
        drawconfusionmatrix.drawMatrix()  # 根据所有predict和label，画出混淆矩阵

    # return correct / total
    return accuracy, f1, cmx


