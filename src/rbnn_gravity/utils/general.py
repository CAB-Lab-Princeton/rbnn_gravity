# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/09/23

import random
import torch
import numpy as np

from PIL import Image


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

def minmax(img: np.ndarray):
    """"""
    min_val = np.min(img) #, axis=(-2, -1))[..., 0]
    max_val = np.max(img) #, axis=(-2, -1))[..., 0]

    img_norm = (img - min_val)/(max_val - min_val)
    return img_norm

def gen_gif(img_array: np.ndarray, save_dir: str):
    """"""
    imgs = [Image.fromarray((minmax(img) * 255).astype('uint8').transpose(1, 2, 0), mode="RGB").resize((128, 128)) for img in img_array[0, ...]]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(save_dir, save_all=True, append_images=imgs[1:], duration=50, loop=0)
