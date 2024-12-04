import random
import torch
from numpy import float16
from torch.distributions import MultivariateNormal, Gamma
from torch.nn import Module
from abc import ABC
from torch import nn


class Perturber(nn.Module):
    arg_cls = None

    def __init__(self, scale: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def change_noise_scale(self, scale):
        self.scale = scale


class DxPrivacy(Perturber):
    def __init__(self, embedder: Module, vocab_size, epsilon: float = 5.0, *args, **kwargs):
        super().__init__(scale=epsilon, *args, **kwargs)
        self.vocab_size = vocab_size
        self.embedder = embedder

    def forward(self, inputs_embeds):
        if self.scale == 0:
            return inputs_embeds
        with torch.no_grad():
            batch_size, seq_len, embed_size = inputs_embeds.shape
            # Sample noise from multivariate normal distribution
            cov_matrix = torch.eye(embed_size).expand(batch_size, seq_len, embed_size, embed_size)
            normal_dist = MultivariateNormal(torch.zeros(embed_size), covariance_matrix=cov_matrix[0, 0])
            noise_v = normal_dist.sample(inputs_embeds.shape[:2])
            norm = torch.linalg.norm(noise_v, dim=-1, keepdim=True)
            norm = torch.where(norm > 0, norm, torch.ones_like(norm))
            noise_v = noise_v / norm
            # Sample scale from gamma distribution
            alpha = embed_size
            # self.scale = relative epsilon = epsilon/embed_size
            beta = self.scale * embed_size
            gamma_dist = Gamma(torch.tensor([alpha]).float(), torch.tensor([beta]))
            scale = gamma_dist.sample()
            # Apply noise
            noise = scale * noise_v
            if inputs_embeds.dtype == float16:
                noise = noise.half()
            inputs_embeds = inputs_embeds + noise.to(inputs_embeds.device)

        all_words = torch.tensor(list([i for i in range(self.vocab_size)])).to(inputs_embeds.device)
        all_embeds = self.embedder(all_words)
        is_half = inputs_embeds.dtype == torch.float16 or all_embeds.dtype == torch.float16
        if is_half:
            inputs_embeds = inputs_embeds.float()
            all_embeds = all_embeds.float()
        cosine_similarities = torch.matmul(inputs_embeds, all_embeds.transpose(0, 1))
        max_token = torch.argmax(cosine_similarities, dim=-1)
        res = all_embeds[max_token]
        if is_half:
            res = res.half()
        return res

def random_choose_noise(input_scales=None, mode='dxp',extra_choices=None):
    dxp_moe_range = {0.08, 0.21, 0.38}
    gaussian_moe_range = {3.0, 5.0, 8.0}
    dc_moe_range = {8.0, 32.0, 64.0}
    if input_scales is None:
        if mode == 'dxp':
            input_scales = dxp_moe_range
        elif mode == 'gaussian':
            input_scales = gaussian_moe_range
        elif mode == 'dc':
            input_scales = dc_moe_range
    scales = set()
    for s in input_scales:
        if s > 0:
            scales.add(s)
    numbers = [random.uniform(min(scales), max(scales)) for _ in
               range(len(scales))]
    numbers += [0, 0, 0, 0]
    plus_one = max(scales) * 2
    if mode == 'dxp' or mode == 'gaussian':
        numbers += [plus_one]
    elif mode == 'dc':
        numbers = list(scales) + [0, 0, 0]
    if extra_choices:
        numbers += extra_choices
    return random.choice(numbers)



def calc_unshift_loss(lm_logits, labels):
    labels = labels.to(lm_logits.device)
    # do not shift
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                            