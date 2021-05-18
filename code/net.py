# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual_dense_block import RDB
from utils import *



BN_MOMENTUM = 0.1
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(channel_in, channel_out, stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out
        
        
class first_Net(nn.Module):
    def __init__(self):
        super(first_Net, self).__init__()
        #pre 
        self.conv01 = nn.Conv2d(4, 16, 3, 1, 1)
        
        self.conv11 = RDB(16,4,16)
        self.conv12 = RDB(16,4,16)
        self.conv13 = RDB(16,4,16)
        
        self.conv20 = RDB(16,4,16)
        self.conv21 = BasicBlock(16,16)
        self.conv22 = BasicBlock(16,3)

    def forward(self, x):
        edge_x = edge_compute(x)
        x = torch.cat((x,edge_x),1)
        
        x01 =  self.conv01(x)
        
        x11 = self.conv11(x01)
        x12 = self.conv12 (x11)
        x13 = self.conv13 (x12)
        x20 = x11 + x12 + x13
        x20 = self.conv20 (x20)
        x21 = self.conv21 (x20)
        x22 = self.conv22 (x21)
        return (x22)
class second_Net(nn.Module):
    def __init__(self):
        super(second_Net, self).__init__()
        #pre 
        self.conv01 = nn.Conv2d(6, 16, 3, 1, 1)
        
        self.conv11 = RDB(16,4,16)
        self.conv12 = RDB(16,4,16)
        self.conv21 = BasicBlock(16,16)
        self.conv22 = BasicBlock(16,3)

    def forward(self, x,y):
        x = torch.cat((x,y),1)
        x01 =  self.conv01(x)
        
        x11 = self.conv11(x01)
        x12 = self.conv12 (x11)

        x21 = self.conv21 (x12)
        x22 = self.conv22 (x21)
        return (x22)

class final_Net(nn.Module):
    def __init__(self):
        super(final_Net, self).__init__()
        self.conv01 = first_Net()
        self.conv02 = second_Net()
    def forward(self, x):
        first_out = self.conv01(x)
        second_out = self.conv02(x,first_out)
        return (first_out,second_out)