from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ST_LSTM import *


class Pred_LSTM(object):
	"""docstring for Pred_LSTM"""
	def __init__(self):
		super(Pred_LSTM, self).__init__()
		self.n_layers = n_layers
		self.hidden_sizes = hidden_sizes
		self.input_sizes = input_sizes
		self.num_steps = num_steps
		self.batch_size = batch_size

		self.M = dict()
		self.C = dict()
		self.H = dict()

		self.embeddings = nn.Embedding(self.vocab_size, self.input_sizes[0])

		self.cells = []
		for i in self.n_layers:
			cell = SpatioTemporal_LSTM(self.hidden_sizes[i], self.input_sizes[i])
			self.cells.append(cell)

	def forward(self, inputs):
		outputs = []

		for n_step in self.num_steps:
			for j,cell in enumerate(self.cells):

				if n_step == 0 and j==0: 			#Initialize state
					self.M[n_step], self.C[n_step], self.H[n_step] = cell(inputs[:,n_step,:], )
					continue

				if n_step == 0:						#Initialize state
					self.M[n_step], self.C[n_step], self.H[n_step] = cell(self.H[n_step-1], )

				if j==0: 							#Input from inputs
					self.M[n_step], self.C[n_step], self.H[n_step] = cell(inputs[:,n_step,:], (self.H[n_step],self.C[n_step],self.M.items()[-1][1]))
					continue

				if j==self.n_layers-1:				#Capture output
					out, self.C[n_step], self.H[n_step] = cell(self.H[n_step-1], (self.H[n_step],self.C[n_step],self.M[n_step-1]))
					outputs.append(out)
					continue

				self.M[n_step], self.C[n_step], self.H[n_step] = cell(self.H[n_step-1], (self.H[n_step],self.C[n_step],self.M[n_step-1]))

		return outputs


