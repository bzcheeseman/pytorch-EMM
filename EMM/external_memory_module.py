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
        self.memory = Variable(torch.ones(memory_banks, *self.memory_dims)) * 1e-4

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
        w = torch.div(w, torch.sum(w).data[0] + 1e-5)

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

        # Apply Batch Norm layer
        self.memory = self.bank_bn(self.memory.permute(1, 0, 2)).permute(1, 0, 2)

        # Decouple histories - clear memory after each run?
        self.wr = Variable(self.wr.data)
        self.ww = Variable(self.ww.data)
        self.memory = Variable(self.memory.data)

        return r_t


class EMM_GPU(nn.Module):
    def __init__(self,
                 num_hidden,
                 read_size,
                 batch_size,
                 memory_banks=1,
                 memory_dims=(128, 20)):
        super(EMM_GPU, self).__init__()

        self.memory_dims = memory_dims
        self.num_hidden = num_hidden
        self.read_size = read_size
        self.batch_size = batch_size
        self.memory_banks = memory_banks

        # Gates for filters ####################################################################################
        self.focus_gate = nn.Linear(self.num_hidden, 1)
        self.wide_gate = nn.Linear(self.num_hidden, 1)

        # Reading ##############################################################################################
        self.focused_read_filter = Variable(torch.ones(1, self.memory_banks, 3, 3))  # (out, in, kh, kw)
        self.wide_read_filter = Variable(torch.ones(1, self.memory_banks, 7, 7))
        self.hidden_to_read_f = nn.Linear(self.num_hidden, self.memory_banks * 3 * 3)
        self.hidden_to_read_w = nn.Linear(self.num_hidden, self.memory_banks * 7 * 7)

        self.conv_focused_to_read = nn.Linear((self.memory_dims[0] - 3 + 1) * (self.memory_dims[1] - 3 + 1),
                                              self.read_size)
        self.conv_wide_to_read = nn.Linear((self.memory_dims[0] - 7 + 1) * (self.memory_dims[1] - 7 + 1),
                                           self.read_size)

        # Writing ##############################################################################################
        self.hidden_focused_to_conv = nn.Linear(self.num_hidden,
                                                (self.memory_dims[0] + 3 - 1)
                                                * (self.memory_dims[1] + 3 - 1))
        self.hidden_wide_to_conv = nn.Linear(self.num_hidden,
                                             (self.memory_dims[0] + 7 - 1)
                                             * (self.memory_dims[1] + 7 - 1))

        # Add and erase gate for writing ##
        self.add_gate = nn.Linear(self.num_hidden, 1)
        self.erase_gate = nn.Linear(self.num_hidden, 1)

        self.focused_write_filter = Variable(torch.ones(self.memory_banks, 1, 3, 3))  # (out, in, kh, kw)
        self.wide_write_filter = Variable(torch.ones(self.memory_banks, 1, 7, 7))
        self.hidden_to_write_f = nn.Linear(self.num_hidden, self.memory_banks * 3 * 3)
        self.hidden_to_write_w = nn.Linear(self.num_hidden, self.memory_banks * 7 * 7)

        # Memory for the external memory module #################################################################
        self.memory = Variable(torch.ones(batch_size, memory_banks, *self.memory_dims)) * 1e-5

    def _read_from_mem(self, gf_t, gw_t, h_t):

        frf_t = Funct.sigmoid(self.hidden_to_read_f(h_t))
        wrf_t = Funct.sigmoid(self.hidden_to_read_w(h_t))

        frf_t = frf_t.view(*self.focused_read_filter.size())
        wrf_t = wrf_t.view(*self.wide_read_filter.size())

        gf_t = gf_t.repeat(*frf_t.size())
        gw_t = gw_t.repeat(*wrf_t.size())

        self.focused_read_filter = frf_t * gf_t + self.focused_read_filter * (1.0 - gf_t)

        self.wide_read_filter = wrf_t * gw_t + self.wide_read_filter * (1.0 - gw_t)

        focused_read = Funct.relu(Funct.conv2d(self.memory, self.focused_read_filter)).squeeze(1)
        # (126, 18) = memory_dims - 3 + 1

        wide_read = Funct.relu(Funct.conv2d(self.memory, self.wide_read_filter)).squeeze(1)
        # (122, 14) = memory_dims - 7 + 1

        focused_read = focused_read.view(-1, num_flat_features(focused_read))
        wide_read = wide_read.view(-1, num_flat_features(wide_read))

        read = Funct.relu(self.conv_focused_to_read(focused_read)) + Funct.relu(self.conv_wide_to_read(wide_read))

        return read

    def _write_to_mem(self, gf_t, gw_t, h_t):

        # compute filter parameters
        fwf_t = Funct.sigmoid(self.hidden_to_write_f(h_t))
        wwf_t = Funct.sigmoid(self.hidden_to_write_w(h_t))

        fwf_t = fwf_t.view(*self.focused_write_filter.size())
        wwf_t = wwf_t.view(*self.wide_write_filter.size())

        gf_t = gf_t.repeat(*fwf_t.size())
        gw_t = gw_t.repeat(*wwf_t.size())

        # Gate the filters
        self.focused_write_filter = fwf_t * gf_t + self.focused_write_filter * (1.0 - gf_t)

        self.wide_write_filter = wwf_t * gw_t + self.wide_write_filter * (1.0 - gw_t)

        # turn h into write matrices
        mwf_t = Funct.relu(self.hidden_focused_to_conv(h_t))
        mww_t = Funct.relu(self.hidden_wide_to_conv(h_t))

        mwf_t = mwf_t.view(1, 1, (self.memory_dims[0] + 3 - 1), (self.memory_dims[1] + 3 - 1))
        mww_t = mww_t.view(1, 1, (self.memory_dims[0] + 7 - 1), (self.memory_dims[1] + 7 - 1))

        # convolve converted h into the correct form
        focused_write = Funct.conv2d(mwf_t, self.focused_write_filter)
        wide_write = Funct.conv2d(mww_t, self.wide_write_filter)

        focused_write = focused_write
        wide_write = wide_write

        # and finally add it to the memory, add or erase
        add = self.add_gate(h_t)
        erase = self.erase_gate(h_t)

        add = add.repeat(*wide_write.size())
        erase = erase.repeat(*focused_write.size())

        m_t = focused_write * (1.0 - erase) + focused_write * add + wide_write * (1.0 - erase) + wide_write * add

        m_t = Funct.softmax(m_t)  # not sure about this

        self.memory = self.memory + m_t

    def forward(self, h_t):  # something is making it so that one of the variables is being modified.
        h_t = h_t.view(-1, num_flat_features(h_t))

        gf_t = Funct.sigmoid(self.focus_gate(h_t))
        gw_t = Funct.sigmoid(self.wide_gate(h_t))

        self._write_to_mem(gf_t, gw_t, h_t)

        r_t = self._read_from_mem(gf_t, gw_t, h_t)

        return r_t


if __name__ == "__main__":
    emm = EMM_GPU(100, 50, 1, memory_banks=5)
    hidden = Variable(torch.rand(1, 100))
    emm.forward(hidden)
