#
# Created by Aman LaChapelle on 3/11/17.
#
# pytorch-NTM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-NTM/LICENSE.txt
#

import torch
from torch.autograd import Variable


def cosine_similarity(x, y, epsilon=1e-6):

    z = torch.mm(x, y.transpose(0, 1))
    z /= torch.sqrt(x.norm(2).pow(2) * y.norm(2).pow(2) + epsilon).data[0]

    return z
