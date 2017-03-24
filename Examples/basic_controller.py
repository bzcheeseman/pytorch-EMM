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
from EMM import EMM


class FeedForwardController(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 batch_size,
                 memory_dims=(128, 20)):

        super(FeedForwardController, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.memory_dims = memory_dims

        self.in_to_hid = nn.Linear(self.num_inputs, self.num_hidden)
        self.read_to_hid = nn.Linear(self.memory_dims[1], self.num_hidden)

    def forward(self, x, read):

        x.contiguous()
        x = x.view(-1, num_flat_features(x))
        read = read.view(-1, self.memory_dims[1])

        hidden = Funct.relu(self.in_to_hid(x) + self.read_to_hid(read), inplace=True)
        return hidden


class NTM(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 batch_size,
                 mem_banks,
                 memory_dims=(128, 20)):
        super(NTM, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_inputs
        self.batch_size = batch_size
        self.mem_banks = mem_banks
        self.memory_dims = memory_dims

        self.hidden = Variable(
            torch.FloatTensor(batch_size, 1, self.num_hidden)
                .normal_(0.0, 1. / self.num_hidden))

        self.EMM = EMM(self.num_hidden, self.batch_size,
                       num_shifts=3, memory_banks=self.mem_banks,
                       memory_dims=self.memory_dims)

        self.controller = FeedForwardController(self.num_inputs, self.num_hidden, self.batch_size,
                                                memory_dims=self.memory_dims)

        self.hid_to_out = nn.Linear(self.num_hidden, self.num_outputs)

    def step(self, x_t, bank_no):

        r_t = self.EMM(self.hidden, bank_no)

        self.hidden = self.controller(x_t, r_t)
        h_t = self.hidden.view(-1, num_flat_features(self.hidden))

        self.hidden = Variable(self.hidden.data)
        out = Funct.hardtanh(self.hid_to_out(h_t), min_val=0.0, max_val=1.0)
        return out

    def forward(self, x):
        x = x.permute(1, 0, 2, 3)

        if self.training:
            outs = torch.stack(
                [
                    self.step(x[t], t % self.mem_banks) for t in np.arange(x.size()[0])
                ], 0)
        else:  # should I change this to be only one bank?
            outs = torch.stack(
                [
                    self.step(x[t], t % self.mem_banks) for t in np.arange(x.size()[0])
                    ], 0)

        outs = outs.permute(1, 0, 2)

        return outs


def generate_copy_data(input_shape, seq_len, num_samples=2e4):
    output = []

    input_tensor = torch.FloatTensor(*input_shape).uniform_(0, 1)

    for j in range(int(num_samples)):
        sample = []
        for i in range(seq_len):
            sample.append(torch.bernoulli(input_tensor))

        sample = torch.cat(sample).view(seq_len, *input_shape)
        output.append(sample.unsqueeze(0))

    output = torch.cat(output, 0)
    return torch.FloatTensor(output), torch.FloatTensor(output)


def train_ntm(batch, num_inputs, seq_len, num_hidden):

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset

    test_data, test_labels = generate_copy_data((num_inputs, 1), seq_len)

    test = TensorDataset(test_data, test_labels)  # TODO: subclass Dataset class to have varying sequence lengths

    data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

    ntm = NTM(num_inputs, num_hidden, batch, mem_banks=3)

    max_epochs = 5

    try:
        ntm.load_state_dict(torch.load("models/copy_seqlen_{}.dat".format(seq_len)))
        max_epochs = 1
    except FileNotFoundError or AttributeError:
        pass

    ntm.train()

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(ntm.parameters(), weight_decay=0.0005)  # weight_decay=0.0005 seems to be a good balance

    for epoch in range(max_epochs):
        running_loss = 0.0

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs)
            labels = Variable(labels)

            ntm.zero_grad()
            outputs = ntm(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % 1000 == 999:
                print('[%d, %5d] average loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

                plottable_input = torch.squeeze(inputs.data[0]).numpy()
                plottable_output = torch.squeeze(outputs.data[0]).numpy()
                plt.imshow(plottable_input)
                plt.savefig("plots/ntm/{}_{}_input.png".format(epoch + 1, i + 1))
                plt.close()
                plt.imshow(plottable_output)
                plt.savefig("plots/ntm/{}_{}_net_output.png".format(epoch + 1, i + 1))
                plt.close()

    torch.save(ntm.state_dict(), "models/copy_seqlen_{}.dat".format(seq_len))
    print("Finished Training")

    data, labels = generate_copy_data((8, 1), 5 * seq_len, 1000)

    test = TensorDataset(data, labels)
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
    train_ntm(1, 8, 20, 100)  # total loss on 5x seq length is 0.017 with 1 memory bank, 1 epoch
                              # total loss on 5x seq length is 0.0xx with 3 memory banks, 1 epoch
