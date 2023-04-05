import torch
import np
import random




def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed) # TODO: is there a better way to set numpy seed? create np_rng here?
    random.seed(seed)


