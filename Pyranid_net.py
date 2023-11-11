from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import scipy.io as sio

# from kmeans_pytorch import kmeans, kmeans_predict

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x


#class ConvMod(nn.Module):
#    def __init__(self,L,act_func):
#        super(ConvMod, self).__init__()
#        self.conlayer=nn.Sequential(nn.Conv2d(L,  L, (1, 1)),
#            nn.InstanceNorm2d( L, affine=True),
#            act_func,)
#    def forward(self,x):
#        return self.conlayer(x)


class ConvFcn(torch.nn.Module):
    def __init__(self, L, out_dim,activation):
        super(ConvFcn, self).__init__()
        self.L = L

        act_func=self.ActivationLayer(activation)
        self.att = ConvMod(out_dim)
#        self.att = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.down1=nn.Sequential(nn.Conv2d(out_dim, L, kernel_size=3, padding=1),
            act_func,
            nn.MaxPool2d(kernel_size=2, stride=2),)
        self.down2 = nn.Sequential(nn.Conv2d(L, 2*L, kernel_size=3, padding=1),
            act_func,
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.down3 = nn.Sequential(nn.Conv2d(2*L, 3*L, kernel_size=3, padding=1),
            act_func,
            nn.MaxPool2d(kernel_size=2, stride=2) )

        self.layers1 = nn.Sequential(
            nn.Conv2d(4 * L + 2 + 28, 4 * L, (1, 1)),
            nn.InstanceNorm2d(4 * L, affine=True),
            act_func, )

        self.layers2 = nn.Sequential(nn.Conv2d(5 * L, 2 * L, (1, 1)),
                                     nn.InstanceNorm2d(2 * L, affine=True),
                                     act_func)

        self.layers3 = nn.Sequential(nn.Conv2d(4 * L, 1 * L, (1, 1)),
                                     nn.InstanceNorm2d(1 * L, affine=True),
                                     act_func,

                                     ConvMod(1 * L), )
        self.layers4 = nn.Sequential(nn.Conv2d(4 * L, 1 * L, (1, 1)),
                                     nn.InstanceNorm2d(1 * L, affine=True),
                                     act_func,

                                     ConvMod(1 * L),
                                     nn.Conv2d(1 * L, out_dim, (1, 1))
                                     )
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=8, mode='nearest')

    def forward(self, x,attachedvec):
        # attachedvec=x[:,130:,:,:]
        # x=x[:,:130,:,:]
        attachedvec=self.add_noise(attachedvec,0.1)
        attachedvec=self.att(attachedvec)
        out1 = self.layers1(torch.cat([x,attachedvec], 1))
        f1=self.down1(attachedvec)
        p1=self.up1(f1)
        out1 = self.add_noise(out1, 0.1)
        out2=self.layers2(torch.cat([out1,p1],1))
        out2 = self.add_noise(out2, 0.1)
        f2 = self.down2(f1)
        p2 = self.up2(f2)
        out3 = self.layers3(torch.cat([out2, p2], 1))
        out3 = self.add_noise(out3, 0.1)
        f3 = self.down3(f2)
        p3 = self.up3(f3)
        out = self.layers4(torch.cat([out3, p3], 1))
        return out


    def add_noise(self,inputs, std_dev):
        # 创建一个与输入形状相同的正态分布噪声张量
        noise = torch.randn_like(inputs) * std_dev
        # 将噪声张量添加到输入层
        noisy_input = inputs + noise
        return noisy_input


#    def add_noise(self,inputs, std_dev):
#        # 创建一个与输入形状相同的正态分布噪声张量
#        noise = torch.randn_like(inputs) * std_dev
#        # 将噪声张量添加到输入层
#        noisy_input = nn.ReLU()(inputs + noise)
#        noisy_input=noisy_input.squeeze(0).cpu().numpy()
#        QE, bit = 0.4, 2048
#        input = np.random.binomial((noisy_input * bit / QE).astype(int), QE)
#        input = np.float32(input) / np.float32(bit)
#        return torch.from_numpy(input).unsqueeze(0).cuda()


    def ActivationLayer(self,act_type):
        if act_type=='relu':
            act_layer=nn.ReLU(inplace=True)
        elif act_type=='leaky_relu':
            act_layer=nn.LeakyReLU(inplace=True)
        elif act_type=='GELU':
            act_layer=nn.GELU()
        elif act_type=='swish':
            act_layer=Swish()
        return act_layer


# used as class:
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
