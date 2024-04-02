import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CHnet(nn.Module):
    def __init__(self, ks=5, in_channels=1, cw=32):
        super(CHnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=cw, kernel_size=ks, bias=True)
        self.conv2 = nn.Conv2d(in_channels=cw, out_channels=cw, kernel_size=ks, bias=True)
        self.conv3 = nn.Conv2d(in_channels=cw, out_channels=2*cw, kernel_size=ks, bias=True)
        self.conv4 = nn.Conv2d(in_channels=2*cw, out_channels=2*cw, kernel_size=ks, bias=True)
        self.conv5 = nn.Conv2d(in_channels=2*cw, out_channels=4*cw, kernel_size=ks, bias=True)
        self.conv6 = nn.Conv2d(in_channels=4*cw, out_channels=4*cw, kernel_size=ks, bias=True)
        self.conv7 = nn.Conv2d(in_channels=4*cw, out_channels=4*cw, kernel_size=ks, bias=True)
        self.conv8 = nn.Conv2d(in_channels=4*cw, out_channels=4*cw, kernel_size=ks, bias=True)
        self.conv9 = nn.Conv2d(in_channels=4*cw, out_channels=4*cw, kernel_size=ks, bias=True)
        self.conv10 = nn.Conv2d(in_channels=4*cw, out_channels=4*cw, kernel_size=ks, bias=True)
        self.linear1 = nn.Conv2d(in_channels=4*cw, out_channels=1024, kernel_size=1, bias=True)
        self.linear2 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, bias=True)
        self.linear3 = nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1, bias=True)

    def forward(self, x):
        p1 = 0.1
        p2 = 0.5
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p1)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout2d(x, p1)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.dropout2d(x, p1)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.dropout2d(x, p1)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.dropout2d(x, p1)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.dropout2d(x, p1)
        x = self.conv7(x)
        x = F.relu(x)
        x = F.dropout2d(x, p1)
        x = self.conv8(x)
        x = F.relu(x)
        x = F.dropout2d(x, p1)
        x = self.conv9(x)
        x = F.relu(x)
        x = F.dropout2d(x, p1)
        x = self.conv10(x)
        x = F.relu(x)
        x = F.dropout2d(x, p1)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p2)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout2d(x, p2)
        x = self.linear3(x)
        return x


