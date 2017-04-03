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
        memory = Variable(torch.rand(*self.memory_dims) * 1e-2, requires_grad=True)

        # Read/write weights
        ww = Variable(torch.eye(self.batch_size, self.memory_dims[0]), requires_grad=True)
        wr = torch.zeros(self.num_reads, self.batch_size, self.memory_dims[0])
        wr[:, 0, 0] = 1.0
        wr = Variable(wr, requires_grad=True)

        return wr, ww, memory

    def _weight_update(self, h_t, w_tm1, m_t):

        h_t = h_t.view(-1, num_flat_features(h_t))

        k_t = torch.clamp(Funct.relu(self.key(h_t)), 0.0, 1.0)  # vector size (batch x memory_dims[1])
        beta_t = Funct.relu(self.beta(h_t))  # batch x number
        g_t = torch.clamp(Funct.hardtanh(self.gate(h_t), min_val=0.0, max_val=1.0), min=0.0, max=1.0)  # batch x number
        s_t = Funct.softmax(self.shift(h_t))  # vector size (batch x num_shifts)
        gamma_t = 1.0 + Funct.relu(self.gamma(h_t))  # batch x number

        # Content Addressing
        beta_tr = beta_t.repeat(1, self.memory_dims[0])
        w_c = Funct.softmax(cosine_similarity(k_t, m_t) * beta_tr)  # vector size (batch x memory_dims[0])

        # Interpolation
        w_g = g_t.expand_as(w_c) * w_c + (1.0 - g_t).expand_as(w_tm1) * w_tm1  # vector size (batch x memory_dims[0])

        # Convolutional Shift
        w_tilde = circular_convolution(w_g, s_t)

        # Sharpening
        w = torch.div(w_tilde.pow(gamma_t.expand_as(w_tilde)),
                      torch.sum(w_tilde.pow(gamma_t.expand_as(w_tilde))).data[0])

        return w

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
    def __init__(self,
                 num_hidden,
                 read_size,
                 batch_size,
                 memory_banks=1,
                 memory_dims=(10, 10)):
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
        self.hidden_to_read_f = nn.Linear(self.num_hidden, self.memory_banks * 1 * 1)
        self.hidden_to_read_w = nn.Linear(self.num_hidden, self.memory_banks * 5 * 5)

        self.conv_focused_to_read = nn.Linear(
            (self.memory_dims[0] - 1 + 1) * (self.memory_dims[1] - 1 + 1), self.read_size
        )
        self.conv_wide_to_read = nn.Linear(
            (self.memory_dims[0] - 5 + 1) * (self.memory_dims[1] - 5 + 1), self.read_size
        )

        # Writing ##############################################################################################
        self.hidden_focused_to_conv = nn.Linear(self.num_hidden,
                                                (self.memory_dims[0] + 1 - 1)
                                                * (self.memory_dims[1] + 1 - 1))
        self.hidden_wide_to_conv = nn.Linear(self.num_hidden,
                                             (self.memory_dims[0] + 5 - 1)
                                             * (self.memory_dims[1] + 5 - 1))

        # Add and erase gate for writing ##
        self.add_gate = nn.Linear(self.num_hidden, 1)
        self.erase_gate = nn.Linear(self.num_hidden, 1)

        self.hidden_to_write_f = nn.Linear(self.num_hidden, self.memory_banks * 1 * 1)
        self.hidden_to_write_w = nn.Linear(self.num_hidden, self.memory_banks * 5 * 5)

    def init_filters_mem(self):
        focused_read_filter = Variable(torch.ones(1, self.memory_banks, 1, 1), requires_grad=True)  # (out, in, kh, kw)
        wide_read_filter = Variable(torch.ones(1, self.memory_banks, 5, 5), requires_grad=True)
        focused_write_filter = Variable(torch.ones(self.memory_banks, 1, 1, 1), requires_grad=True)  # (out, in, kh, kw)
        wide_write_filter = Variable(torch.ones(self.memory_banks, 1, 5, 5), requires_grad=True)
        memory = Variable(torch.ones(self.batch_size, self.memory_banks, *self.memory_dims), requires_grad=True) * 1e-5

        return focused_read_filter, wide_read_filter, focused_write_filter, wide_write_filter, memory

    def _read_from_mem(self, gf_t, gw_t, h_t, frf, wrf, m):

        frf_t = Funct.hardtanh(self.hidden_to_read_f(h_t), min_val=0.0, max_val=1.0)
        wrf_t = Funct.hardtanh(self.hidden_to_read_w(h_t), min_val=0.0, max_val=1.0)

        frf_t = frf_t.view(*frf.size())
        wrf_t = wrf_t.view(*wrf.size())

        frf_t = frf_t * gf_t.expand_as(frf_t) \
                                   + frf * (1.0 - gf_t.expand_as(frf_t))

        wrf_t = wrf_t * gw_t.expand_as(wrf_t) \
                                + wrf * (1.0 - gw_t.expand_as(wrf_t))

        focused = Funct.relu(Funct.conv2d(m, frf_t))
        # (126, 18) = memory_dims - 3 + 1
        #
        wide = Funct.relu(Funct.conv2d(m, wrf_t))
        # (122, 14) = memory_dims - 5 + 1

        focused_read = focused.view(-1, num_flat_features(focused))
        wide_read = wide.view(-1, num_flat_features(wide))

        # problem right here - tensor is somehow 4D on the backward pass, but only when it comes through the controller
        read = self.conv_wide_to_read(wide_read) + Funct.relu(self.conv_focused_to_read(focused_read))

        return read, frf_t, wrf_t

    def _write_to_mem(self, gf_t, gw_t, h_t, fwf, wwf, m):

        # Compute filter parameters
        fwf_t = Funct.hardtanh(self.hidden_to_write_f(h_t), min_val=0.0, max_val=1.0)
        wwf_t = Funct.hardtanh(self.hidden_to_write_w(h_t), min_val=0.0, max_val=1.0)

        fwf_t = fwf_t.view(*fwf.size())
        wwf_t = wwf_t.view(*wwf.size())

        # Gate the filters
        fwf_t = fwf_t * gf_t.expand_as(fwf_t) + \
                                    fwf * (1.0 - gf_t.expand_as(fwf))

        wwf_t = wwf_t * gw_t.expand_as(wwf_t) + \
                                 wwf * (1.0 - gw_t.expand_as(wwf))

        # Turn h into write matrices
        mwf_t = Funct.relu(self.hidden_focused_to_conv(h_t))
        mww_t = Funct.relu(self.hidden_wide_to_conv(h_t))

        # Convolve converted h into the correct form
        focused_write = Funct.conv2d(
            mwf_t.view(1, 1, (self.memory_dims[0] + 1 - 1), (self.memory_dims[1] + 1 - 1)),
            fwf_t
        )
        wide_write = Funct.conv2d(
            mww_t.view(1, 1, (self.memory_dims[0] + 5 - 1), (self.memory_dims[1] + 5 - 1)),
            wwf_t
        )

        # And finally write it to the memory, add or erase
        add = torch.clamp(Funct.relu(self.add_gate(h_t)), 0.0, 1.0)
        erase = torch.clamp(Funct.hardtanh(self.erase_gate(h_t), min_val=0.0, max_val=1.0), 0.0, 1.0)

        mem_write = focused_write * (1.0 - erase.expand_as(focused_write)) + add.expand_as(focused_write) + \
              wide_write * (1.0 - erase.expand_as(wide_write)) + add.expand_as(wide_write)

        m_t = m + mem_write
        return m_t, fwf_t, wwf_t

    def forward(self, h_t, frf, wrf, fwf, wwf, m):
        h_t = h_t.view(-1, num_flat_features(h_t))

        gf_t = torch.clamp(1.2 * Funct.sigmoid(self.focus_gate(h_t)) - 0.1, min=0.0, max=1.0)
        gw_t = torch.clamp(1.2 * Funct.sigmoid(self.wide_gate(h_t)) - 0.1, min=0.0, max=1.0)

        r_t, frf_t, wrf_t = self._read_from_mem(gf_t, gw_t, h_t, frf, wrf, m)

        m_t, fwf_t, wwf_t = self._write_to_mem(gf_t, gw_t, h_t, fwf, wwf, m)

        return r_t, frf_t, wrf_t, fwf_t, wwf_t, m_t

