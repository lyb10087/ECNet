import numpy as np
from openpyxl import load_workbook, Workbook
import torch.utils.data as dataf
import torch.nn as nn
from scipy import io
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import time
import utils
from Houston.svd.svd import svd_gather
from vit_pytorch import ViT


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SIAM(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)

        self.gate_treshold = gate_treshold

        self.sigomid = nn.Sigmoid()

    def forward(self, S):
        gn_S = self.gn(S)
        w_gamma = F.softmax(self.gn.gamma, dim=0)

        reweigts = self.sigomid(gn_S * w_gamma)

        info_mask = w_gamma > self.gate_treshold
        noninfo_mask = w_gamma <= self.gate_treshold
        S_1 = info_mask * reweigts * S
        S_2 = noninfo_mask * reweigts * S
        S = self.reconstruct(S_1, S_2)
        return S

    def reconstruct(self, S_1, S_2):
        S_11, S_12 = torch.split(S_1, S_1.size(1) // 2, dim=1)
        S_21, S_22 = torch.split(S_2, S_2.size(1) // 2, dim=1)
        return torch.cat([S_11 + S_22, S_12 + S_21], dim=1)


class CRRM(nn.Module):

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.rich_channel = rich_channel = int(alpha * op_channel)
        self.redundant_channel = redundant_channel = op_channel - rich_channel
        self.squeeze1 = nn.Conv2d(rich_channel, rich_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(redundant_channel, redundant_channel // squeeze_radio, kernel_size=1, bias=False)

        self.GWC = nn.Conv2d(rich_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(rich_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(redundant_channel // squeeze_radio, op_channel - redundant_channel // squeeze_radio,
                              kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, I):
        rich, redundant = torch.split(I, [self.rich_channel, self.redundant_channel], dim=1)
        rich, redundant = self.squeeze1(rich), self.squeeze2(redundant)

        Y1 = self.GWC(rich) + self.PWC1(rich)
        Y2 = torch.cat([self.PWC2(redundant), redundant], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class BMFM(nn.Module):
    # Balanced Modal Fusion Module(BMFM)
    def __init__(self,
                 op_channel: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SIAM = SIAM(op_channel,
                         group_num=group_num,
                         gate_treshold=gate_treshold)
        self.CRRM = CRRM(op_channel,
                         alpha=alpha,
                         squeeze_radio=squeeze_radio,
                         group_size=group_size,
                         group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SIAM(x)
        x = self.CRRM(x)

        return x


DataPath1 = './Houston2013/HSI.mat'
DataPath2 = './Houston2013/LiDAR_1.mat'
TRPath = './Houston2013/TRLabel.mat'
TSPath = './Houston2013/TSLabel.mat'


def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


same_seeds(0)
patchsize1 = 11
patchsize2 = 11
batchsize = 64
EPOCH = 200
LR = 0.001
NC = 20  # Dimensionality reduction of hyperspectral data to 20 channels

Classes=15

FM=32
class ECNet(nn.Module):
    def __init__(self):
        super(ECNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=NC,
                out_channels=FM,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM * 2, 3, 1, 1),
            nn.BatchNorm2d(FM * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

        )
        self.out1 = nn.Sequential(

            nn.Linear(FM * 4, Classes),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=FM,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(FM),  # BN can improve the accuracy by 4%-5%
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

        )

        self.out2 = nn.Sequential(

            nn.Linear(FM * 4, Classes),
        )

        self.out3 = nn.Sequential(

            nn.Linear(FM * 4, Classes),
        )

        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1 / 3]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1 / 3]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([1 / 3]))
        self.SSAM = ViT(image_size=5,
                       near_band=1,
                       num_patches=128,
                       num_classes=15,
                       dim=64,
                       depth=5,
                       heads=4,
                       mlp_dim=8,
                       dropout=0.1,
                       emb_dropout=0.1,
                       mode='ViT')
        self.BMFM = BMFM(op_channel=FM * 2)
        self.svd= svd_gather

    def forward(self, x1, x2):
        x1 = self.svd(x1, 102)[0]

        x1 = self.conv1(x1)

        x1_tf = x1

        out1 = self.SSAM(x1_tf.reshape(-1, 32, 25))

        x1 = self.conv2(x1)
        x1 = self.BMFM(x1)
        x1 = self.conv3(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.conv4(x2)
        x2 = self.conv2(x2)
        x2 = self.BMFM(x2)
        x2 = self.conv3(x2)

        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)

        x = x1 + x2

        out3 = self.out3(x)
        out = self.coefficient1 * out1 + self.coefficient2 * out2 + self.coefficient3 * out3

        return out1, out2, out3, out


