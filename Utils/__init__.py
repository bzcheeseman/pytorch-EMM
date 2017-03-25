#
# Created by Aman LaChapelle on 3/23/17.
#
# pytorch-EMM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-EMM/LICENSE.txt
#

__all__ = ["flat_features", "similarities"]

from .flatfeatures import num_flat_features
from .similarities import cosine_similarity
from .tasks import CopyTask
