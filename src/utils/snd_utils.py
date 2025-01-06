import torch
from torch.distributions.gamma import Gamma

def sample_noise_Chi(d_shape, eta):
    n_dim = d_shape[-1]
    alpha = torch.ones(d_shape) * n_dim
    beta = torch.ones(d_shape) * eta
    m = Gamma(alpha, beta)
    l_lst = m.sample()
    v_lst = -2 * torch.rand(d_shape) + 1
    noise = l_lst * v_lst
    noise = noise
    return noise