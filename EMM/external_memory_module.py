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


class EMM_NTM(nn.Module):
    def __init__(self,
                 num_hidden,
                 batch_size,
                 num_reads=1,
                 num_shifts=3,
                 memory_banks=1,
                 memory_dims=(128, 20)):
        super(EMM_NTM, self).__init__()

        self.memory_dims = memory_dims
        self.memory_banks = memory_banks
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_shifts = num_shifts
        self.num_reads = num_reads

        # Memory for the external memory module
        self.memory = Variable(torch.rand(memory_banks, *self.memory_dims)) * 1e-5

        # Batch normalization across the memory banks - forces them together
        self.bank_bn = nn.BatchNorm1d(memory_banks)

        # Read/write weights
        self.ww = Variable(torch.rand(self.batch_size, self.memory_dims[0]))
        self.ww = Funct.softmax(self.ww)

        self.wr = Variable(torch.rand(num_reads, self.batch_size, self.memory_dims[0]))
        self.wr = Funct.softmax(self.wr)

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
        w = w_tilde.pow(gamma_tr)
        w = Funct.softmax(w)

        return w

    def _write_to_mem(self, h_t, w_tm1, m_t):
        h_t = h_t.view(-1, num_flat_features(h_t))

        e_t = Funct.hardtanh(self.hid_to_erase(h_t), min_val=0.0, max_val=1.0)
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

    def clear_mem(self):
        self.memory = Variable(torch.ones(self.memory_banks, *self.memory_dims)) * 1e-5

    def forward(self, h_t, bank_no):
        # Update write weights and write to memory
        self.ww = self._weight_update(h_t, self.ww, self.memory[bank_no].clone())
        self.memory[bank_no] = self._write_to_mem(h_t, self.ww, self.memory[bank_no].clone())

        # Update read weights and read from memory
        r_t = []
        for i in range(self.num_reads):
            self.wr[i] = self._weight_update(h_t, self.wr[i].clone(), self.memory[bank_no].clone())
            r_t.append(self._read_from_mem(h_t, self.wr[i].clone(), self.memory[bank_no].clone()))

        r_t = torch.cat(r_t, 1)

        # print(self.memory.size())

        # Apply Batch Norm layers
        # self.memory[bank_no] = self.mem_bn(self.memory[bank_no])
        # self.memory = self.bank_bn(self.memory.permute(1, 0, 2)).permute(1, 0, 2)

        # Decouple histories - clear memory after each run?
        self.wr = Variable(self.wr.data)
        self.ww = Variable(self.ww.data)
        self.memory = Variable(self.memory.data)

        return r_t


class EMM_GPU(nn.Module):
    def __init__(self,
                 num_hidden,
                 batch_size,
                 memory_banks=1,
                 memory_dims=(128, 20)):
        super(EMM_GPU, self).__init__()

        self.memory_dims = memory_dims
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.memory_banks = memory_banks

        # transform from something useful for the controller - check the sizes, want it to end up the size of the memory
        self.lin_1_out = nn.Linear(self.num_hidden, self.memory_dims[0] * self.memory_dims[1])
        self.lin_3_out = nn.Linear(self.num_hidden, (self.memory_dims[0] - 3 + 1) * (self.memory_dims[1] - 3 + 1))
        self.lin_5_out = nn.Linear(self.num_hidden, (self.memory_dims[0] - 5 + 1) * (self.memory_dims[1] - 5 + 1))
        self.lin_7_out = nn.Linear(self.num_hidden, (self.memory_dims[0] - 7 + 1) * (self.memory_dims[1] - 7 + 1))
        # filters into the full memory banks
        self.filter_1_in = nn.Conv2d(1, memory_banks, 1)
        self.filter_3_in = nn.Conv2d(1, memory_banks, 3)
        self.filter_5_in = nn.Conv2d(1, memory_banks, 5)
        self.filter_7_in = nn.Conv2d(1, memory_banks, 7)

        # filters out of the full memory banks
        self.filter_1_out = nn.Conv2d(memory_banks, 1, 1)
        self.filter_3_out = nn.Conv2d(memory_banks, 1, 3)
        self.filter_5_out = nn.Conv2d(memory_banks, 1, 5)
        self.filter_7_out = nn.Conv2d(memory_banks, 1, 7)
        # transform to something useful for the controller
        self.lin_1_out = nn.Linear(self.memory_dims[0]*self.memory_dims[1], self.num_hidden)
        self.lin_3_out = nn.Linear((self.memory_dims[0]-3+1)*(self.memory_dims[1]-3+1), self.num_hidden)
        self.lin_5_out = nn.Linear((self.memory_dims[0] - 5 + 1) * (self.memory_dims[1] - 5 + 1), self.num_hidden)
        self.lin_7_out = nn.Linear((self.memory_dims[0] - 7 + 1) * (self.memory_dims[1] - 7 + 1), self.num_hidden)

        # Memory for the external memory module
        self.memory = Variable(torch.ones(batch_size, memory_banks, *self.memory_dims)) * 1e-5

    def _read_from_mem(self):
        # Read from the memory
        single = Funct.hardtanh(self.filter_1_out(self.memory))  # collapses it into one matrix
        triple = Funct.hardtanh(self.filter_3_out(self.memory))  # also condenses it down
        quint = Funct.hardtanh(self.filter_5_out(self.memory))  # condenses it more
        sept = Funct.hardtanh(self.filter_7_out(self.memory))  # condenses it the most

        single.contiguous()
        triple.contiguous()
        quint.contiguous()
        sept.contiguous()
        single = single.view(-1, num_flat_features(single))
        triple = triple.view(-1, num_flat_features(triple))
        quint = quint.view(-1, num_flat_features(quint))
        sept = sept.view(-1, num_flat_features(sept))

        single = Funct.relu(self.lin_1_out(single), inplace=True)
        triple = Funct.relu(self.lin_3_out(triple), inplace=True)
        quint = Funct.relu(self.lin_5_out(quint), inplace=True)
        sept = Funct.relu(self.lin_7_out(sept), inplace=True)

        # combine these into one somehow? Just concat?


    def _write_to_mem(self, h_t):
        pass
        # do the same process as read in reverse


if __name__ == "__main__":
    emm = EMM_GPU(100, 1, memory_banks=5)
    emm._read_from_mem()

