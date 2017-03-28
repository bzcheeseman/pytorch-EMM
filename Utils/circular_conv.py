#
# Created by Aman LaChapelle on 3/27/17.
#
# pytorch-EMM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-EMM/LICENSE.txt
#


import torch
from torch.autograd import Variable
import numpy as np
import math
from scipy.linalg import circulant

def circular_convolution(v, k):
    size = v.size()[1]  # the first dimension is the batch size
    kernel_size = k.size()[1]
    kernel_shift = int(math.floor(kernel_size / 2.0))

    # print(k)

    circ = circulant(np.arange(size)).T[np.arange(-(kernel_shift % 2), (kernel_shift % 2) + 1)][::-1]
    k = k.unsqueeze(2).repeat(1, 1, size)
    windows = Variable(torch.from_numpy(v.data.numpy()[:, circ]))
    out = torch.sum(k + windows, dim=1)
    return out.squeeze(1)
