import torch
import opt


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
    x = x.transpose(1, 3)
    return x

def gamma_correct(x, gamma=1.5, exposure=1):
    return torch.clamp((x ** gamma) * exposure, 0, 1)

def levels(x, black, white):
    return torch.clamp((x - black)/(white-black), 0, 1)
