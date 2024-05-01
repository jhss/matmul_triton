import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Step1 (Naive)
load(name='naive_step1', sources=[''])
