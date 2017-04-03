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
import torch.optim as optim
import numpy as np

from Utils import num_flat_features
from EMM import EMM_NTM, EMM_GPU
from Utils import CopyTask


class FeedForwardController(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 batch_size,
                 num_reads=1,
                 memory_dims=(128, 20)):

        super(FeedForwardController, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.memory_dims = memory_dims

        self.in_to_hid = nn.Linear(self.num_inputs, self.num_hidden)
        self.read_to_hid = nn.Linear(self.memory_dims[1]*num_reads, self.num_hidden)

    def forward(self, x, read):

        x = x.contiguous()
        x = x.view(-1, num_flat_features(x))
        read = read.contiguous()
        read = read.view(-1, num_flat_features(read))

        x = Funct.relu(self.in_to_hid(x) + self.read_to_hid(read))

        return x


class GRUController(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 batch_size,
                 num_reads=1,
                 memory_dims=(128, 20)):

        super(GRUController, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.memory_dims = memory_dims

        self.gru = nn.GRUCell(
            input_size=self.num_inputs,
            hidden_size=self.num_hidden
        )

        self.read_to_in = nn.Linear(self.memory_dims[1]*num_reads, self.num_inputs)

    def forward(self, x, read, h_t):
        x = x.contiguous()
        r = Funct.relu(self.read_to_in(read))
        r = r.view(*x.size())
        x = x + r
        x = x.view(-1, num_flat_features(x))
        h_tp1 = self.gru(x, h_t)

        return h_tp1


class NTM(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_outputs,
                 batch_size,
                 num_reads,
                 memory_dims=(128, 20)):
        super(NTM, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.num_reads = num_reads
        self.memory_dims = memory_dims

        self.hidden = Variable(torch.zeros(batch_size, self.num_hidden))

        self.EMM = EMM_NTM(self.num_hidden, self.batch_size, num_reads=self.num_reads,
                           num_shifts=3, memory_dims=self.memory_dims)

        self.controller = FeedForwardController(self.num_inputs, self.num_hidden, self.batch_size,
                                                num_reads=self.num_reads, memory_dims=self.memory_dims)

        self.hid_to_out = nn.Linear(self.num_hidden, self.num_outputs)

        self.wr, self.ww, self.memory = self.EMM.init_weights_mem()

    def forward(self, x):

        x = x.permute(1, 0, 2, 3)

        def step(x_t):
            r_t, self.wr, self.ww, self.memory = self.EMM(self.hidden, self.wr, self.ww, self.memory)

            h_t = self.controller(x_t, r_t)
            h_t = h_t.view(-1, num_flat_features(h_t))
            self.hidden = h_t

            out = Funct.sigmoid(self.hid_to_out(self.hidden))
            return out

        outs = torch.stack(
            [
                step(x_t) for x_t in torch.unbind(x, 0)
            ], 1)  # stack along the correct index so we don't have to permute.

        self.hidden = Variable(self.hidden.data)  # decouple history here
        self.wr = Variable(self.wr.data)
        self.ww = Variable(self.ww.data)
        self.memory = Variable(self.memory.data)
        return outs


class GPU_NTM(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_outputs,
                 batch_size,
                 mem_banks,
                 num_reads,
                 memory_dims=(8, 8)):
        super(GPU_NTM, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.mem_banks = mem_banks
        self.num_reads = num_reads
        self.memory_dims = memory_dims

        self.EMM = EMM_GPU(self.num_hidden, self.num_reads*self.memory_dims[1], self.batch_size,
                           memory_banks=self.mem_banks, memory_dims=self.memory_dims)

        self.controller = FeedForwardController(self.num_inputs, self.num_hidden, self.batch_size,
                                        num_reads=self.num_reads, memory_dims=self.memory_dims)

        self.hid_to_out = nn.Linear(self.num_hidden, self.num_outputs)

    def init_hidden(self):
        focused_read_filter, \
        wide_read_filter, \
        focused_write_filter, \
        wide_write_filter, \
        memory = self.EMM.init_filters_mem()
        hidden = Variable(torch.zeros(self.batch_size, self.num_hidden))

        return hidden, focused_read_filter, wide_read_filter, focused_write_filter, wide_write_filter, memory

    def forward(self, x, h, frf, wrf, fwf, wwf, m):

        x = x.permute(1, 0, 2, 3)

        def step(x_t, h_t, frf_t, wrf_t, fwf_t, wwf_t, m_t):

            r_t, frf_t, wrf_t, fwf_t, wwf_t, m_t = self.EMM(h_t, frf_t, wrf_t, fwf_t, wwf_t, m_t)

            h_t = self.controller(x_t, r_t)

            out = Funct.sigmoid(self.hid_to_out(h_t))

            return out, h_t, frf_t, wrf_t, fwf_t, wwf_t, m_t

        x_t = torch.unbind(x, 0)
        out = []
        for i in range(x.size()[0]):
            o, h, frf, wrf, fwf, wwf, m = step(x_t[i], h, frf, wrf, fwf, wwf, m)
            out.append(o)

        outs = torch.stack(out, 1)

        return outs


def train_gpu(batch, num_inputs, seq_len, num_hidden):
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    ntm = GPU_NTM(num_inputs, num_hidden, num_inputs, batch, num_reads=1, mem_banks=5)

    try:
        ntm.load_state_dict(torch.load("models/copy_seqlen_{}.dat".format(seq_len)))
    except FileNotFoundError or AttributeError:
        pass

    ntm.train()
    h, frf, wrf, fwf, wwf, m = ntm.init_hidden()

    criterion = nn.L1Loss()
    optimizer = optim.RMSprop(ntm.parameters(), lr=1e-3, weight_decay=0.00001)

    max_seq_len = 5  # change the training schedule to be curriculum training
    for length in range(2, max_seq_len):
        running_loss = 0.0

        test = CopyTask(length, [num_inputs, 1], num_samples=2e4)

        data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

        for epoch in range(5):

            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                inputs = Variable(inputs)
                labels = Variable(labels)

                ntm.zero_grad()
                outputs = ntm(inputs, h, frf, wrf, fwf, wwf, m)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]

                if i % 1000 == 999:
                    print('[length: %d, epoch: %d, i: %5d] average loss: %.3f' % (length, epoch + 1, i + 1,
                                                                                  running_loss / 1000))

                    plt.imshow(m[0, 0].data.numpy())
                    plt.savefig("plots/ntm/{}_{}_{}_memory.png".format(length, epoch + 1, i + 1))
                    plt.close()
                    plottable_input = torch.squeeze(inputs.data[0]).numpy()
                    plottable_output = torch.squeeze(outputs.data[0]).numpy()
                    plottable_true_output = torch.squeeze(labels.data[0]).numpy()
                    plt.imshow(plottable_input)
                    plt.savefig("plots/ntm/{}_{}_{}_input.png".format(length, epoch + 1, i + 1))
                    plt.close()
                    plt.imshow(plottable_output)
                    plt.savefig("plots/ntm/{}_{}_{}_net_output.png".format(length, epoch + 1, i + 1))
                    plt.close()
                    plt.imshow(plottable_true_output)
                    plt.savefig("plots/ntm/{}_{}_{}_true_output.png".format(length, epoch + 1, i + 1))
                    plt.close()

                    if running_loss / 1000 <= 0.005:
                        break

                    running_loss = 0.0

    torch.save(ntm.state_dict(), "models/gpu_copy_seqlen_{}.dat".format(seq_len))
    print("Finished Training")

    test = CopyTask(5 * max_seq_len, [num_inputs - 1, 1], num_samples=1e4)
    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

    total_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs.volatile = True
        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = ntm(inputs)

        if i % 1000 / batch == (1000 / batch) - 1:
            plottable_input = torch.squeeze(inputs.data[0]).numpy()
            plottable_output = torch.squeeze(outputs.data[0]).numpy()
            plt.imshow(plottable_input)
            plt.savefig("plots/ntm/{}_{}_input_test.png".format(epoch + 1, i + 1))
            plt.close()
            plt.imshow(plottable_output)
            plt.savefig("plots/ntm/{}_{}_net_output_test.png".format(epoch + 1, i + 1))
            plt.close()

        total_loss += len(data) * criterion(outputs, labels).data

    print("Total Loss: {}".format(total_loss / len(data_loader)))


def train_ntm(batch, num_inputs, seq_len, num_hidden):

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    ntm = NTM(num_inputs, num_hidden, num_inputs, batch, num_reads=1)

    try:
        ntm.load_state_dict(torch.load("models/copy_seqlen_{}.dat".format(seq_len)))
    except FileNotFoundError or AttributeError:
        pass

    ntm.train()

    state = ntm.state_dict()

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(ntm.parameters(), lr=1e-3, weight_decay=0.0005)
    # weight_decay=0.0005 seems to be a good balance

    max_seq_len = 20  # change the training schedule to be curriculum training
    for length in range(2, max_seq_len):
        running_loss = 0.0

        test = CopyTask(length, [num_inputs, 1], num_samples=2e4)

        data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

        for epoch in range(5):

            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                inputs = Variable(inputs)
                labels = Variable(labels)

                ntm.zero_grad()
                outputs = ntm(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # assert not (ntm.state_dict()['hid_to_out.bias'] == state['hid_to_out.bias'])[0]

                running_loss += loss.data[0]

                if i % 1000 == 999:

                    print('[length: %d, epoch: %d, i: %5d] average loss: %.3f' % (length, epoch + 1, i + 1,
                                                                                  running_loss / 1000))

                    plt.imshow(ntm.wr.squeeze(0).data.numpy())
                    plt.savefig("plots/ntm/{}_{}_{}_read.png".format(length, epoch+1, i + 1))
                    plt.close()
                    plt.imshow(ntm.memory.squeeze().data.numpy().T)
                    plt.savefig("plots/ntm/{}_{}_{}_memory.png".format(length, epoch + 1, i + 1))
                    plt.close()
                    plt.imshow(ntm.ww.data.numpy())
                    plt.savefig("plots/ntm/{}_{}_{}_write.png".format(length, epoch + 1, i + 1))
                    plt.close()
                    plottable_input = torch.squeeze(inputs.data[0]).numpy()
                    plottable_output = torch.squeeze(outputs.data[0]).numpy()
                    plottable_true_output = torch.squeeze(labels.data[0]).numpy()
                    plt.imshow(plottable_input)
                    plt.savefig("plots/ntm/{}_{}_{}_input.png".format(length, epoch+1, i + 1))
                    plt.close()
                    plt.imshow(plottable_output)
                    plt.savefig("plots/ntm/{}_{}_{}_net_output.png".format(length, epoch+1, i + 1))
                    plt.close()
                    plt.imshow(plottable_true_output)
                    plt.savefig("plots/ntm/{}_{}_{}_true_output.png".format(length, epoch+1, i + 1))
                    plt.close()

                    if running_loss/1000 <= 0.005:
                        break

                    running_loss = 0.0

    torch.save(ntm.state_dict(), "models/copy_seqlen_{}.dat".format(seq_len))
    print("Finished Training")

    test = CopyTask(5 * max_seq_len, [num_inputs-1, 1], num_samples=1e4)
    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

    total_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs.volatile = True
        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = ntm(inputs)

        if i % 1000/batch == (1000/batch)-1:
            plottable_input = torch.squeeze(inputs.data[0]).numpy()
            plottable_output = torch.squeeze(outputs.data[0]).numpy()
            plt.imshow(plottable_input)
            plt.savefig("plots/ntm/{}_{}_input_test.png".format(epoch + 1, i + 1))
            plt.close()
            plt.imshow(plottable_output)
            plt.savefig("plots/ntm/{}_{}_net_output_test.png".format(epoch + 1, i + 1))
            plt.close()

        total_loss += len(data) * criterion(outputs, labels).data

    print("Total Loss: {}".format(total_loss / len(data_loader)))

if __name__ == '__main__':
    # train_ntm(1, 8, 5, 100)
    train_gpu(1, 8, 5, 100)

    # OLD WAY OF SEQUENCING
    # total loss on 5x seq length is 0.017 with 1 memory bank, 1 epoch
    # total loss on 5x seq length is 7.81e-5 with 3 memory banks, 1 epoch
