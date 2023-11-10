# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/09/23

import random
import torch
import numpy as np


# Reproducibility Utility Functions
def setup_reproducibility(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)