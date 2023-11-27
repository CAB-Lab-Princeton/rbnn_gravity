# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/09/23

import random
import torch
import numpy as np


# Reproducibility Utility Functions
def setup_reproducibility(seed: int):
    """"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_reproducibility_hd(seed: int, cnn_reprod: bool = True):
    """"""
    if cnn_reprod:
        # Set benchmark to be constant and deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Seed all other sources of randomness
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)