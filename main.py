from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ST_LSTM import *


class Encoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self):
		super(Encoder, self).__init__()
		self.n_layers = n_layers
		self.hidden_sizes = hidden_sizes
		self.input_sizes = input_sizes

		self.M = dict()
		self.C = dict()
		self.H = dict()

		self.h = dict()
        self.c = dict()
        self.m = { 0 : Parameter(torch.Tensor())}


        self.h[0] = Parameter(torch.Tensor(shape))
        self.h[1] = Parameter(torch.Tensor(shape))
        self.h[2] = Parameter(torch.Tensor(shape))
        self.h[3] = Parameter(torch.Tensor(shape))


        self.c[0] = Parameter(torch.Tensor(shape))
        self.c[1] = Parameter(torch.Tensor(shape))
        self.c[2] = Parameter(torch.Tensor(shape))
        self.c[3] = Parameter(torch.Tensor(shape))

		self.cells = nn.ModuleList([])
		for i in self.n_layers:
			cell = SpatioTemporal_LSTM(self.hidden_sizes[i], self.input_sizes[i])
			self.cells.append(cell)

		self._reset_parameters()

	def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


	def forward(self, input_, first_timestep = False):
		for j,cell in enumerate(self.cells):
			if first_timestep == True:
				if j == 0:
					self.H[j], self.C[j], self.M[j] = cell(input_, (self.h[j],self.c[j],self.m[j]))
					continue
				else:
					self.H[j], self.C[j], self.M[j] = cell(self.H[j-1], (self.h[j],self.c[j],self.M[j-1]))
				continue

			if j==0:
				self.H[j], self.C[j], self.M[j] = cell(input_, (self.H[j],self.C[j],self.M[self.n_layers-1]))
				continue

			self.H[j], self.C[j], self.M[j] = cell(self.H[j-1],(self.H[j],self.C[j],self.M[j-1]))

		return self.H , self. C, self.M

	def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size)) #################SHAPE
        if use_cuda:
            return result.cuda()
        else:
            return result


class Decoder(nn.Module):
	"""
	docstring for Decoder
	
	Using M in zigzag fashion as suggested in Spatiotemporal LSTM
	
	"""
	def __init__(self):
		super(Decoder, self).__init__()
		self.n_layers = n_layers
		self.hidden_sizes = hidden_sizes
		self.input_sizes = input_sizes


		self.cells = nn.ModuleList([])
		for i in self.n_layers:
			cell = SpatioTemporal_LSTM(self.hidden_sizes[i], self.input_sizes[i])
			self.cells.append(cell)

	def forward(self, input_, C,H,M):
		for j,cell in enumerate(self.cells):
			if j==0:
				H[j], C[j],M[j] = cell(input_,(H[j],C[j],M[n_layers-1]))

			if j==n_layers-1:
				H[j], C[j],M[j] = cell(H[j-1],(H[j],C[j],M[j-1]))
				output = H[j]
		return output

	def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
