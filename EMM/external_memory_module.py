#
# Created by Aman LaChapelle on 3/23/17.
#
# pytorch-EMM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-EMM/LICENSE.txt
#

import torch
import torch.nn as nn
import torch.nn.functional as Funct
from torch.autograd import Variable
import numpy as np

from Utils import cosine_similarity
from Utils import circular_convolution
from Utils import num_flat_features


class EMM_NTM(nn.Module):
    def __init__(self,
                 num_hidden,
                 batch_size,
                 num_reads=1,
                 num_shifts=3,
                 memory_dims=(128, 20)):
        super(EMM_NTM, self).__init__()

        self.memory_dims = memory_dims
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_shifts = num_shifts
        self.num_reads = num_reads

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

    def init_weights_mem(self):
        # Memory for the external memory module
        memory = Variable(torch.rand(*self.memory_dims) * 1e-2)

        # Read/write weights
        ww = Variable(torch.eye(self.batch_size, self.memory_dims[0]))
        wr = torch.zeros(self.num_reads, self.batch_size, self.memory_dims[0])
        wr[:, 0, 0] = 1.0
        wr = Variable(wr)

        return wr, ww, memory

    def _weight_update(self, h_t, w_tm1, m_t):

        h_t = h_t.view(-1, num_flat_features(h_t))

        k_t = torch.clamp(Funct.relu(self.key(h_t)), 0.0, 1.0)  # vector size (batch x memory_dims[1])
        beta_t = Funct.relu(self.beta(h_t))  # batch x number
        g_t = torch.clamp(Funct.hardtanh(self.gate(h_t), min_val=0.0, max_val=1.0), min=0.0, max=1.0)  # batch x number
        s_t = Funct.softmax(self.shift(h_t))  # vector size (batch x num_shifts)
        gamma_t = 1.0 + Funct.sigmoid(self.gamma(h_t))  # batch x number

        # Content Addressing
        beta_tr = beta_t.repeat(1, self.memory_dims[0])
        w_c = Funct.softmax(cosine_similarity(k_t, m_t) * beta_tr)  # vector size (batch x memory_dims[0])

        # Interpolation
        w_g = g_t.expand_as(w_c) * w_c + (1.0 - g_t).expand_as(w_tm1) * w_tm1  # vector size (batch x memory_dims[0])

        # Convolutional Shift
        w_tilde = circular_convolution(w_g, s_t)

        # Sharpening
        w = (w_tilde + 1e-4).pow(gamma_t.expand_as(w_tilde))
        w = w / (torch.sum(w).expand_as(w) + 1)

        return w_tilde

    def _write_to_mem(self, h_t, w_tm1, m_t):
        h_t = h_t.view(-1, num_flat_features(h_t))

        e_t = torch.clamp(Funct.hardtanh(self.hid_to_erase(h_t), min_val=0.0, max_val=1.0), min=0.0, max=1.0)
        a_t = torch.clamp(Funct.relu(self.hid_to_add(h_t)), min=0.0, max=1.0)

        mem_erase = Variable(torch.zeros(*m_t.size()))
        mem_add = Variable(torch.zeros(*m_t.size()))

        for i in range(e_t.size()[0]):
            mem_erase += torch.ger(w_tm1[i], e_t[i])
            mem_add += torch.ger(w_tm1[i], a_t[i])

        m_t = m_t * (1.0 - mem_erase) + mem_add

        return m_t

    def _read_from_mem(self, h_t, w_tm1, m_t):
        r_t = torch.mm(w_tm1, m_t)
        return r_t

    def forward(self, h_t, wr, ww, m):
        # Update write weights and write to memory
        ww_t = self._weight_update(h_t, ww, m)
        m_t = self._write_to_mem(h_t, ww, m)

        # Update read weights and read from memory
        wr_t = torch.stack(
            [
                self._weight_update(h_t, w, m) for w in torch.unbind(wr, 0)
            ], 0
        )

        r_t = torch.stack(
            [
                self._read_from_mem(h_t, w, m) for w in torch.unbind(wr, 0)
            ], 1
        )

        return r_t.squeeze(1), wr_t, ww_t, m_t  # batch_size x num_reads


class EMM_GPU(nn.Module):
    # want to write a sequence to memory, then read it out should convolve h directly?
    # treat memory like NGPU? write (embed) and then run through cgru unit, read, and take cgru to next time step?
    # or softmax over memory banks to see which one to use
    def __init__(self,
                 num_hidden,
                 read_size,
                 batch_size,
                 memory_banks=32,
                 memory_dims=(8, 8)):
        super(EMM_GPU, self).__init__()

        self.memory_dims = memory_dims
        self.num_hidden = num_hidden
        self.read_size = read_size
        self.batch_size = batch_size
        self.memory_banks = memory_banks

        self.hidden_conv = nn.Sequential(  # will need to unsqueeze hidden
            nn.Conv1d(1, 32, 7, padding=3),
            nn.MaxPool1d(2, stride=2),  # downsample by 2
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.MaxPool1d(2, stride=2),  # downsample by 2
            nn.ReLU()
        )

        # Choosing a Memory Bank/Program Bank ##################################################################
        self.which_bank = nn.Linear(64 * int(self.num_hidden/4), self.memory_banks)

        # Weights for reading/writing ##################################################################################
        self.read_weights = nn.Linear(64 * int(self.num_hidden/4), self.memory_dims[0])
        self.write_weights = nn.Linear(64 * int(self.num_hidden / 4), self.memory_dims[0])

        # Gates ################################################################################################
        self.read_gate = nn.Linear(self.num_hidden, 1)
        self.write_gate = nn.Linear(self.num_hidden, 1)
        self.mem_gate = nn.Linear(self.num_hidden, 1)

        # Reading ##############################################################################################
        self.read = nn.Linear(self.memory_dims[1], self.read_size)

        # Writing ##############################################################################################
        self.conv_to_erase = nn.Linear(64 * int(self.num_hidden/4), self.memory_dims[1])
        self.conv_to_add = nn.Linear(64 * int(self.num_hidden/4), self.memory_dims[1])

        # self.mem_expand = nn.Conv2d(1, self.memory_banks, 3, padding=1)  # should I deal with this differently?

    def init_weights_mem(self):
        wr = Variable(torch.eye(1, self.memory_dims[0]))
        ww = Variable(torch.eye(1, self.memory_dims[0]))
        memory = Variable(torch.rand(self.memory_banks, *self.memory_dims)) * 1e-2
        return wr, ww, memory

    def _read_from_mem(self, wr_t, m_j):

        r_t = torch.mv(m_j, wr_t.squeeze()).unsqueeze(0)
        r_t = Funct.relu(self.read(r_t))

        return r_t, wr_t

    def _write_to_mem(self, hp, ww_t, m_j):

        a_t = Funct.relu(self.conv_to_add(hp))
        e_t = Funct.relu(self.conv_to_erase(hp))

        mem_erase = torch.ger(ww_t.squeeze(), e_t.squeeze())
        mem_add = torch.ger(ww_t.squeeze(), a_t.squeeze())

        m_j = (1.0 - mem_erase) + mem_add
        m_j = m_j.unsqueeze(0)
        return m_j, ww_t

    def forward(self, h_t, wr, ww, m):
        h_t = h_t.view(-1, num_flat_features(h_t))

        gr_t = torch.clamp(1.2 * Funct.sigmoid(self.read_gate(h_t)) - 0.1, min=0.0, max=1.0)
        gw_t = torch.clamp(1.2 * Funct.sigmoid(self.write_gate(h_t)) - 0.1, min=0.0, max=1.0)
        gm_t = torch.clamp(0.6 * Funct.tanh(self.mem_gate(h_t)), min=-0.5, max=0.5)

        h_t = h_t.unsqueeze(1)
        hp = torch.clamp(1.2 * Funct.sigmoid(self.hidden_conv(h_t)) - 0.1, min=0.0, max=1.0)
        hp = hp.view(-1, num_flat_features(hp))

        # Calc memory bank
        bank = Funct.softmax(1.2 * Funct.sigmoid(self.which_bank(hp)) - 0.1)
        bank = bank.unsqueeze(0).transpose(2, 0).repeat(1, self.memory_dims[0], self.memory_dims[1])
        m_j = bank * m
        m_j = m_j.sum(0).squeeze(0)

        # Calc read weights
        wr_t = Funct.softmax(self.read_weights(hp))
        wr_t = gr_t.expand_as(wr_t) * wr_t + (1.0 - gr_t.expand_as(wr)) * wr

        # Read
        r_t, wr_t = self._read_from_mem(wr_t, m_j)

        # Calc write weights
        ww_t = Funct.softmax(self.write_weights(hp))
        ww_t = gw_t.expand_as(ww_t) * ww_t + (1.0 - gw_t.expand_as(ww)) * ww

        m_jt, ww_t = self._write_to_mem(hp, ww_t, m_j)
        # print(bank)
        m_t = m_jt.repeat(self.memory_banks, 1, 1)
        m_t = m_t * bank
        m_t = (0.5 + gm_t.expand_as(m_t)) * m_t + (0.5 - gm_t.expand_as(m)) * m  # is this right?

        return r_t, m_t, wr_t, ww_t

