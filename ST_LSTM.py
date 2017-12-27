from __future__ import print_function


import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

import math

class SpatioTemporal_LSTM(nn.Module):
	"""docstring for SpatioTemporal_LSTM"""
	def __init__(self):
		super(SpatioTemporal_LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size

		self.linear = nn.linear(self.input_size + self.hidden_size , 4*self.hidden_size)
		

		self.weight_xg = Parameter(torch.Tensor())
		self.weight_hg = Parameter(torch.Tensor())
		self.weight_xi = Parameter(torch.Tensor())
		self.weight_hi = Parameter(torch.Tensor())
		self.weight_xf = Parameter(torch.Tensor())
		self.weight_hf = Parameter(torch.Tensor())
		self.weight_xg_ = Parameter(torch.Tensor())
		self.weight_mg = Parameter(torch.Tensor())
		self.weight_xi_ = Parameter(torch.Tensor())
		self.weight_mi = Parameter(torch.Tensor())
		self.weight_xf_ = Parameter(torch.Tensor())
		self.weight_mf = Parameter(torch.Tensor())
		self.weight_xo = Parameter(torch.Tensor())
		self.weight_ho = Parameter(torch.Tensor())
		self.weight_co = Parameter(torch.Tensor())
		self.weight_mo = Parameter(torch.Tensor())
		self.weight_1x1 = Parameter(torch.Tensor(1,1))

		if bias:
            self.bias_g = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_i = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_f = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_g_ = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_i_ = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_f_ = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_o = Parameter(torch.Tensor(4 * hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _compute_cell(self, x, c, h, M):
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

    	return M , h , c


	def forward(self,input, state=None):
		if state is None:
			h = Variable(torch.Tensor())
			c = Variable(torch.Tensor())
			state = (c,h)

		







		




		
		


"""

class LSTMCell(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size=hidden_size
        self.lin = nn.Linear( input_size+hidden_size , 4*hidden_size )
         
    def forward(self, x, state0):
        h0,c0=state0
        x_and_h0 = torch.cat((x,h0), 1)
        u=self.lin(x_and_h0)
        i=F.sigmoid( u[ : , 0*self.hidden_size : 1*self.hidden_size ] )
        f=F.sigmoid( u[ : , 1*self.hidden_size : 2*self.hidden_size ] )
        g=F.tanh(    u[ : , 2*self.hidden_size : 3*self.hidden_size ] )
        o=F.sigmoid( u[ : , 3*self.hidden_size : 4*self.hidden_size ] )
        c= f*c0 + i*g
        h= o*F.tanh(c)
        return (h,c)


"""