import sys
sys.path.append("..")

from .backbone import *

import torch
import torch.nn as nn

import math
import numpy as np


class LSTM0(nn.Module):
    def __init__(self, in_c=5, ngf=32, k=1):
        super(LSTM0, self).__init__()
        self.conv_gx_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=k, padding=k//2)
        self.conv_ix_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=k, padding=k//2)
        self.conv_ox_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=k, padding=k//2)

    def forward(self, xt):
        """
        :param xt:      bz * 5(num_class) * 240 * 240
        :return:
            hide_1:    bz * ngf(32) * 240 * 240
            cell_1:    bz * ngf(32) * 240 * 240
        """
        gx = self.conv_gx_lstm0(xt)
        ix = self.conv_ix_lstm0(xt)
        ox = self.conv_ox_lstm0(xt)

        gx = torch.tanh(gx)
        ix = torch.sigmoid(ix)
        ox = torch.sigmoid(ox)

        cell_1 = torch.tanh(gx * ix)
        hide_1 = ox * cell_1
        return cell_1, hide_1


class LSTM(nn.Module):
    def __init__(self, in_c=5, ngf=32, k=1):
        super(LSTM, self).__init__()
        self.conv_ix_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=k, padding=k//2, bias=True)
        self.conv_ih_lstm = nn.Conv2d(ngf, ngf, kernel_size=k, padding=k//2, bias=False)

        self.conv_fx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=k, padding=k//2, bias=True)
        self.conv_fh_lstm = nn.Conv2d(ngf, ngf, kernel_size=k, padding=k//2, bias=False)

        self.conv_ox_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=k, padding=k//2, bias=True)
        self.conv_oh_lstm = nn.Conv2d(ngf, ngf, kernel_size=k, padding=k//2, bias=False)

        self.conv_gx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=k, padding=k//2, bias=True)
        self.conv_gh_lstm = nn.Conv2d(ngf, ngf, kernel_size=k, padding=k//2, bias=False)

    def forward(self, xt, cell_t_1, hide_t_1):
        """
        :param xt:          bz * (5+32) * 240 * 240
        :param hide_t_1:    bz * ngf(32) * 240 * 240
        :param cell_t_1:    bz * ngf(32) * 240 * 240
        :return:
        """
        gx = self.conv_gx_lstm(xt)         # output: bz * ngf(32) * 240 * 240
        gh = self.conv_gh_lstm(hide_t_1)   # output: bz * ngf(32) * 240 * 240
        g_sum = gx + gh
        gt = torch.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)          # output: bz * ngf(32) * 240 * 240
        oh = self.conv_oh_lstm(hide_t_1)    # output: bz * ngf(32) * 240 * 240
        o_sum = ox + oh
        ot = torch.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)              # output: bz * ngf(32) * 240 * 240
        ih = self.conv_ih_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
        i_sum = ix + ih
        it = torch.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)              # output: bz * ngf(32) * 240 * 240
        fh = self.conv_fh_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
        f_sum = fx + fh
        ft = torch.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt        # bz * ngf(32) * 240 * 240
        hide_t = ot * torch.tanh(cell_t)            # bz * ngf(32) * 240 * 240

        return cell_t, hide_t

    
class UNet2D(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, degree=64):
        super(UNet2D, self).__init__()

        chs = []
        for i in range(5):
            chs.append((2 ** i) * degree)

        self.downLayer1 = ConvBlock2d(in_ch, chs[0])
        self.downLayer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[0], chs[1]))

        self.downLayer3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[1], chs[2]))

        self.downLayer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[2], chs[3]))

        self.bottomLayer = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[3], chs[4]))

        self.upLayer1 = UpBlock2d(chs[4], chs[3])
        self.upLayer2 = UpBlock2d(chs[3], chs[2])
        self.upLayer3 = UpBlock2d(chs[2], chs[1])
        self.upLayer4 = UpBlock2d(chs[1], chs[0])
#         self.outLayer = nn.Conv2d(chs[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        :param x:   4D Tensor    BatchSize * 4(modal) * W * H
        :return:    4D Tensor    BatchSize * 2        * W * H
        """
        x1 = self.downLayer1(x)     # degree(32)   * 16    * W    * H
        x2 = self.downLayer2(x1)    # degree(64)   * 16/2  * W/2  * H/2
        x3 = self.downLayer3(x2)    # degree(128)  * 16/4  * W/4  * H/4
        x4 = self.downLayer4(x3)    # degree(256)  * 16/8  * W/8  * H/8
        
        x5 = self.bottomLayer(x4)   # degree(512)  * 16/16 * W/16 * H/16
        x = self.upLayer1(x5, x4)   # degree(256)  * 16/8 * W/8 * H/8
        x = self.upLayer2(x, x3)    # degree(128)  * 16/4 * W/4 * H/4
        x = self.upLayer3(x, x2)    # degree(64)   * 16/2 * W/2 * H/2
        x = self.upLayer4(x, x1)    # degree(32)   * 16   * W   * H
#         x = self.outLayer(x)        # out_ch(2 )   * 16   * W   * H
        return x


class Unet_solo(nn.Module):
    def __init__(self, input_nc=1, output_nc=5, ngf=32, k=1, tanh=True):
        super(Unet_solo, self).__init__()
        self.unet = UNet2D(in_ch=input_nc, out_ch=ngf, degree=ngf)
        self.uout = nn.Conv2d(ngf, output_nc, kernel_size=k, stride=1, padding=k//2)
        self.tanh = tanh

    def forward(self, x):
        """
        :param x:  4D tensor    bz * ch * 240 * 240
        :return:
        """
        x = self.unet(x)
        
        if self.tanh:
            return torch.tanh(self.uout(x))
        else:
            return self.uout(x)
    
class Unet_loop(nn.Module):
    def __init__(self, input_nc=1, output_nc=5, ngf=32, k=1, temporal=1, tanh=True):
        super(Unet_loop, self).__init__()
        self.unet = UNet2D(in_ch=input_nc, out_ch=ngf, degree=ngf)
        
        self.uout = nn.Conv2d(ngf, output_nc, kernel_size=k, stride=1, padding=k//2)
        self.temporal = temporal
        self.tanh = tanh

    def forward(self, x):
        """
        :param x:  4D tensor    bz * ch * 240 * 240
        :return:
        """
        u_output = []
        for _ in range(self.temporal):
            x = self.unet(x)
            x = self.uout(x)
            u_output.append(x)
        
        if self.tanh:
            return torch.tanh(x), torch.tanh(torch.stack(u_output, dim=1))
        else:
            return x, torch.stack(u_output, dim=1)
        
        
class Unet_rnn(nn.Module):
    def __init__(self, input_nc=1, output_nc=5, ngf=32, k=1, temporal=1, tanh=True):
        super(Unet_rnn, self).__init__()
        self.unet = UNet2D(in_ch=input_nc, out_ch=ngf, degree=ngf)
        
        self.uout = nn.Conv2d(ngf, output_nc, kernel_size=k, stride=1, padding=k//2)
#         self.uhid = nn.Conv2d(ngf, output_nc, kernel_size=k, stride=1, padding=k//2)
        
        self.uhid = UNet2D(in_ch=input_nc, out_ch=ngf, degree=ngf)
        self.hout = nn.Conv2d(ngf, output_nc, kernel_size=k, stride=1, padding=k//2)
        
        self.temporal = temporal
        self.tanh = tanh

    def forward(self, x):
        """
        :param x:  4D tensor    bz * ch * 240 * 240
        :return:
        """
        u_output = []     
        h = torch.ones(x.shape).double().to(x.device)
        for _ in range(self.temporal):
            x = self.unet(x * h)
            x = self.uout(x)
            u_output.append(x)
            h = torch.sigmoid(self.hout(self.uhid(h)))
        
        if self.tanh:
            return torch.tanh(x), torch.tanh(torch.stack(u_output, dim=1))
        else:
            return x, torch.stack(u_output, dim=1)


class LSTM_Unet(nn.Module):
    def __init__(self, input_nc=1, output_nc=5, ngf=32, temporal=3, k=1, tanh=True):
        super(LSTM_Unet, self).__init__()
        self.temporal = temporal
        self.unet = UNet2D(in_ch=1, out_ch=ngf, degree=ngf)
        self.lstm0 = LSTM0(in_c=output_nc , ngf=ngf)
        self.lstm = LSTM(in_c=output_nc , ngf=ngf)

        self.uout = nn.Conv2d(ngf, output_nc, kernel_size=k, stride=1, padding=k//2)
        self.lout = nn.Conv2d(ngf, output_nc, kernel_size=k, stride=1, padding=k//2)
        
        self.tanh = tanh

    def forward(self, x):
        """
        :param x:  5D tensor    bz * temporal * 4 * 240 * 240
        :return:
        """
        l_output = []
        u_output = []
        cell = None
        hide = None
        for t in range(self.temporal):
            im_t = x[:, t, :, :, :]                # bz * 4 * 240 * 240
            u_last = self.unet(im_t)                # bz * 32 * 240 * 240
            out_t = self.uout(u_last)              # bz * 5 * 240 * 240
            u_output.append(out_t)
            lstm_in = torch.cat((out_t, u_last), dim=1) # bz * 37 * 240 * 240

            if t == 0:
                cell, hide = self.lstm0(lstm_in)   # bz * ngf(32) * 240 * 240
            else:
                cell, hide = self.lstm(lstm_in, cell, hide)
            out_t = self.lout(hide)
            l_output.append(out_t)
        
        if self.tanh:
            return torch.tanh(torch.stack(l_output, dim=1)), torch.tanh(torch.stack(u_output, dim=1))
        else:
            return torch.stack(l_output, dim=1), torch.stack(u_output, dim=1) 

    def predict(self, x):
        """
        :param x:  5D tensor    bz * temporal * 4 * 240 * 240
        :return:
        """
        l_output = []
        u_output = []
        cell = None
        hide = None
        im_t = x # bz * 4 * 240 * 240
        for t in range(self.temporal):   
            u_last = self.unet(im_t)               # bz * 32 * 240 * 240
            out_t = self.uout(u_last)              # bz * 5 * 240 * 240
            u_output.append(out_t)
            lstm_in = torch.cat((out_t, u_last), dim=1) # bz * 37 * 240 * 240

            if t == 0:
                cell, hide = self.lstm0(lstm_in)   # bz * ngf(32) * 240 * 240
            else:
                cell, hide = self.lstm(lstm_in, cell, hide)
            out_t = self.lout(hide)
            im_t = torch.tanh(out_t)
            l_output.append(im_t)
        
        if self.tanh:
            return torch.stack(l_output, dim=1), torch.tanh(torch.stack(u_output, dim=1))
        else:
            return torch.stack(l_output, dim=1), torch.stack(u_output, dim=1) 


    
# class LSTM_Unet_Forward(nn.Module):
#     def __init__(self, input_nc=1, output_nc=5, ngf=32, temporal=3, k=1):
#         super(LSTM_Unet_Forward, self).__init__()
#         self.temporal = temporal
#         self.unet = UNet2D(in_ch=1, out_ch=ngf, degree=ngf)
#         self.lstm0 = LSTM0(in_c=output_nc , ngf=ngf)
#         self.lstm = LSTM(in_c=output_nc , ngf=ngf)

#         self.uout = nn.Conv2d(ngf, output_nc, kernel_size=k, stride=1, padding=k//2)
#         self.lout = nn.Conv2d(ngf, output_nc, kernel_size=k, stride=1, padding=k//2)

#     def forward(self, x):
#         """
#         :param x:  5D tensor    bz * temporal * 4 * 240 * 240
#         :return:
#         """
#         l_output = []
#         u_output = []
#         cell = None
#         hide = None
#         im_t = x # bz * 4 * 240 * 240
#         for t in range(self.temporal):   
#             u_last = self.unet(im_t)               # bz * 32 * 240 * 240
#             out_t = self.uout(u_last)              # bz * 5 * 240 * 240
#             u_output.append(out_t)
#             lstm_in = torch.cat((out_t, u_last), dim=1) # bz * 37 * 240 * 240

#             if t == 0:
#                 cell, hide = self.lstm0(lstm_in)   # bz * ngf(32) * 240 * 240
#             else:
#                 cell, hide = self.lstm(lstm_in, cell, hide)
#             out_t = self.lout(hide)
#             im_t = out_t
#             l_output.append(im_t)
        
#         return torch.stack(l_output, dim=1), torch.stack(u_output, dim=1)