from __future__ import print_function


import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

import math

class SpatioTemporal_LSTM(nn.Module):
    """docstring for SpatioTemporal_LSTM"""
    def __init__(self, hidden_size, input_size):
        super(SpatioTemporal_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.linear = nn.linear(self.input_size + self.hidden_size , 4*self.hidden_size)

        # shape = [shape]

        self.weight_xg = Parameter(torch.Tensor(shape))
        self.weight_hg = Parameter(torch.Tensor(shape))
        self.weight_xi = Parameter(torch.Tensor(shape))
        self.weight_hi = Parameter(torch.Tensor(shape))
        self.weight_xf = Parameter(torch.Tensor(shape))
        self.weight_hf = Parameter(torch.Tensor(shape))
        self.weight_xg_ = Parameter(torch.Tensor(shape))
        self.weight_mg = Parameter(torch.Tensor(shape))
        self.weight_xi_ = Parameter(torch.Tensor(shape))
        self.weight_mi = Parameter(torch.Tensor(shape))
        self.weight_xf_ = Parameter(torch.Tensor(shape))
        self.weight_mf = Parameter(torch.Tensor(shape))
        self.weight_xo = Parameter(torch.Tensor(shape))
        self.weight_ho = Parameter(torch.Tensor(shape))
        self.weight_co = Parameter(torch.Tensor(shape))
        self.weight_mo = Parameter(torch.Tensor(shape))
        self.weight_1x1 = Parameter(torch.Tensor(1,1))

        if bias:
            self.bias_g = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_i = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_f = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_g_ = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_i_ = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_f_ = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_o = Parameter(torch.Tensor(4 * hidden_size))        


        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _compute_cell(self, x, h, c, M):
        g = torch.tanh(F.conv2d(x,self.weight_xg) + F.conv2d(h, self.weight_hg) + self.bias_g)
        i = torch.sigmoid(F.conv2d(x, self.weight_xi) + F.conv2d(h, self.weight_hi) + self.bias_i)
        f = torch.sigmoid(F.conv2d(x, self.weight_xf) + F.conv2d(h, self.weight_hf) + self.bias_f)

        c = f*c + i*g

        g_ =  torch.tanh(F.conv2d(x,self.weight_xg_) + F.conv2d(M, self.weight_mg) + self.bias_g_)
        i_ = torch.sigmoid(F.conv2d(x, self.weight_xi_) + F.conv2d(M, self.weight_mi) + self.bias_i_)
        f = torch.sigmoid(F.conv2d(x, self.weight_xf_) + F.conv2d(M, self.weight_mf) + self.bias_f_)

        M = f_*M + i_*g_

        o = torch.sigmoid(F.conv2d(x, self.weight_xo) + F.conv2d(M, self.weight_mo) + F.conv2d(c, self.weight_co) + F.conv2d(h, self.weight_ho) + self.bias_o)

        h = o * torch.tanh(F.conv2d(torch.cat((c,M), dim= ),self.weight_1x1))

        return h,c,M


    def forward(self,input_, state=None):
        if state is None:
            raise ValueError('nfnaiszfv vsknv')

        h,c,M = state

        cell_output = self._compute_cell(input_,h,c,M)

        return cell_output