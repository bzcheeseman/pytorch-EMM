#
# Created by Aman LaChapelle on 3/23/17.
#
# pytorch-EMM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-EMM/LICENSE.txt
#

import torch
torch.backends.cudnn.libpaths.append("/usr/local/cuda/lib")  # give torch the CUDNN library location
import torch.nn as nn
import torch.nn.functional as Funct
from torch.autograd import Variable
import numpy as np

from Utils import cosine_similarity
from Utils import num_flat_features


class EMM(nn.Module):
    def __init__(self,
                 num_hidden,
                 batch_size,
                 num_shifts=3,
                 memory_banks=1,
                 memory_dims=(128, 20)):
        super(EMM, self).__init__()

        self.memory_dims = memory_dims
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_shifts = num_shifts

        # Memory for the external memory module
        self.memory = Variable(torch.rand(memory_banks, *self.memory_dims)) * 1e-5

        # Batch normalization within a memory bank
        self.mem_bn = nn.BatchNorm1d(self.memory_dims[1])

        # Batch normalization across the memory banks - forces them together
        self.bank_bn = nn.BatchNorm1d(self.memory_dims[0])

        # Read/write weights
        self.w = Variable(torch.eye(self.batch_size, self.memory_dims[0]))

        # Key - Clipped Linear or Relu
        self.key = nn.Linear(self.num_hidden, self.memory_dims[1])

        # Beta - Relu
        self.beta = nn.Linear(self.num_hidden, 1)

        # Gate - Hard Sigmoid
        self.gate = nn.Linear(self.num_hidden, 1)

        # Shift - Softmax
        self.shift = nn.Linear(self.num_hidden, self.num_shifts)

        # Gamma - 1 + Relu
        self.gamma = nn.Linear(self.num_hidden, 1)

        # Erase - Hard Sigmoid
        self.hid_to_erase = nn.Linear(self.num_hidden, self.memory_dims[1])

        # Add - Clipped Linear
        self.hid_to_add = nn.Linear(self.num_hidden, self.memory_dims[1])

    def _weight_update(self, h_t, w_tm1, m_t):

        h_t = h_t.view(-1, num_flat_features(h_t))

        k_t = torch.clamp(self.key(h_t), 0.0, 1.0)  # vector size (memory_dims[1])
        beta_t = Funct.relu(self.beta(h_t))  # number
        g_t = torch.clamp(Funct.hardtanh(self.gate(h_t), min_val=0.0, max_val=1.0), min=0.0,
                          max=1.0)  # number
        s_t = Funct.softmax(self.shift(h_t))  # vector size (num_shifts)
        gamma_t = 1.0 + Funct.relu(self.gamma(h_t))  # number

        batch_size = k_t.size()[0]

        # Content Addressing
        beta_tr = beta_t.repeat(1, self.memory_dims[0])  # problem is here, beta is not changing?
        w_c = Funct.softmax(cosine_similarity(k_t, m_t) * beta_tr)  # vector size (memory_dims[0])

        # Interpolation
        g_tr = g_t.repeat(1, self.memory_dims[0])
        w_g = g_tr * w_c + (1.0 - g_tr) * w_tm1  # vector size (memory_dims[0]) (i think)

        # Convolutional Shift
        conv_filter = s_t.unsqueeze(1).unsqueeze(2)
        w_g_padded = w_g.unsqueeze(1).unsqueeze(2)
        pad = (self.num_shifts // 2, (self.num_shifts - 1) // 2)

        conv = Funct.conv2d(w_g_padded, conv_filter, padding=pad)

        w_tilde = conv[:batch_size, 0, 0, :].contiguous()
        w_tilde = w_tilde.view(batch_size, self.memory_dims[0])

        # Sharpening
        gamma_tr = gamma_t.repeat(1, self.memory_dims[0])
        w = (w_tilde + 1e-6).pow(gamma_tr)
        w /= torch.sum(w, dim=1).repeat(1, self.memory_dims[0])

        return w

    def _write_to_mem(self, h_t, w_tm1, m_t):
        h_t = h_t.view(-1, num_flat_features(h_t))

        e_t = Funct.hardtanh(self.hid_to_erase(h_t), min_val=0.0, max_val=1.0, inplace=True)
        a_t = torch.clamp(Funct.relu(self.hid_to_add(h_t)), min=0.0, max=1.0)

        mem_erase = torch.zeros(*m_t.size())
        mem_add = torch.zeros(*m_t.size())

        for i in range(e_t.size()[0]):  # batch size
            mem_erase += torch.ger(w_tm1[i].data, e_t[i].data)
            mem_add += torch.ger(w_tm1[i].data, a_t[i].data)

        m_tp1 = m_t.data * (1.0 - mem_erase) + mem_add

        return Variable(m_tp1)

    def _read_from_mem(self, h_t, w_tm1, m_t):
        r_t = torch.mm(w_tm1, m_t)
        return r_t

    def forward(self, h_t, bank_no):
        # read from memory
        r_t = self._read_from_mem(h_t, self.w, self.memory[bank_no])
        # TODO: Give r_t to the controller here, use synth. gradients (or similar) to decouple updates?

        # Update the weights
        self.w = self._weight_update(h_t, self.w, self.memory[bank_no])

        # Write to memory
        self.memory[bank_no] = torch.clamp(self._write_to_mem(h_t, self.w, self.memory[bank_no]), 0.0, 1.0)

        # Apply Batch Norm layers
        self.memory[bank_no] = torch.clamp(self.mem_bn(self.memory[bank_no]), 0.0, 1.0)
        self.memory = torch.clamp(self.bank_bn(self.memory), 0.0, 1.0)

        # Decouple histories
        self.w = Variable(self.w.data)
        self.memory = Variable(self.memory.data)

        return r_t
