"""Adaptive mean pooling with deterministic CUDA backward."""

import torch


def mean_bins(x, count):
    length = x.shape[-1]
    if length % count == 0:
        return x.unflatten(-1, (count, length // count)).mean(-1)
    return torch.stack([x[..., i*length//count:((i+1)*length+count-1)//count].mean(-1)
                        for i in range(count)], dim=-1)
