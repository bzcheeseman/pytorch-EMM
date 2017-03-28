#
# Created by Aman LaChapelle on 3/25/17.
#
# pytorch-EMM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-EMM/LICENSE.txt
#

import torch
from torch.utils.data import Dataset
import numpy as np


class CopyTask(Dataset):
    def __init__(self, max_seq_len, input_size, num_samples):
        self.max_seq_len = max_seq_len
        self.input_size = input_size
        self.num_samples = num_samples

        self.input_tensor = torch.FloatTensor(*input_size).uniform_(0, 1)
        self.input_tensor = torch.cat([self.input_tensor, torch.zeros(1, 1)], 0)
        self.input_size[0] += 1

    def __getitem__(self, index):
        sample = []
        sample_label = []

        rand_seq_len = np.random.randint(low=3, high=self.max_seq_len)
        zeros = torch.zeros(*self.input_size)

        sample_label.append(zeros)
        for i in range(rand_seq_len):
            sample.append(torch.bernoulli(self.input_tensor))

        sample.append(zeros)
        for i in range(rand_seq_len):
            sample_label.append(sample[i])

        sample = torch.stack(sample).view(rand_seq_len + 1, *self.input_size)
        sample_label = torch.cat(sample_label).view(rand_seq_len + 1, *self.input_size)

        return sample, sample_label

    def __len__(self):
        return int(self.num_samples)

