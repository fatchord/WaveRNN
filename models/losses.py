import numpy as np
import math
import torch
from torch.distributions.normal import Normal


def gaussian_loss(y_hat, y, log_std_min=-7.0):
    assert y_hat.dim() == 3
    assert y_hat.size(2) == 2
    mean = y_hat[:, :, :1]
    log_std = torch.clamp(y_hat[:, :, 1:], min=log_std_min)
    # TODO: replace with pytorch dist
    log_probs = -0.5 * (- math.log(2.0 * math.pi) - 2. * log_std - torch.pow(y - mean, 2) * torch.exp((-2.0 * log_std)))
    return log_probs.squeeze().mean()


def sample_from_gaussian(y_hat, log_std_min=-7.0, scale_factor=1.0):
    assert y_hat.size(2) == 2
    mean = y_hat[:, :, :1]
    log_std = torch.clamp(y_hat[:, :, 1:], min=log_std_min)
    dist = Normal(mean, torch.exp(log_std))
    sample = dist.sample()
    sample = torch.clamp(torch.clamp(sample, min=-scale_factor), max=scale_factor)
    del dist
    return sample