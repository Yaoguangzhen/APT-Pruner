import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def lookup_latency(lut, mask):
    n = int(torch.sum(mask != 0))
    if n == 0:
        return 0
    else:
        return lut[n - 1]


def estimate_latency(mha_lut, ffn_lut, head_mask, neuron_mask):
    num_hidden_layers = head_mask.shape[0]
    total = 0
    for i in range(num_hidden_layers):
        total += lookup_latency(mha_lut, head_mask[i])
        total += lookup_latency(ffn_lut, neuron_mask[i])
    return total


class PiecewiseLinearLatency:
    def __init__(self, threshold=None, c=None, slope=None):
        self.threshold = threshold
        self.c = c
        self.slope = slope


def fit_latency_fn(lut):
    lut = np.asarray(lut)
    latency_fn = PiecewiseLinearLatency()

    min_error = 10000
    for threshold in range(1, len(lut) + 1):
        c = lut[:threshold].sum() / threshold
        y = lut[threshold:] - c
        x = np.arange(1, len(y) + 1)

        if threshold == len(lut):
            slope = 0
        else:
            slope = (x * y).sum() / (x * x).sum()
        slope = 0 if slope < 0 else slope

        approximated = [c] * threshold
        for i in range(1, len(lut) - threshold + 1):
            approximated.append(slope * i + c)
        approximated = np.asarray(approximated)

        squared_error = ((lut - approximated) * (lut - approximated)).sum()
        if squared_error < min_error:
            min_error = squared_error
            latency_fn.threshold = threshold
            latency_fn.c = c
            latency_fn.slope = slope

    return latency_fn


def prune_tokens_vit(model, token_mask, mha_lut, ffn_lut):
    input_tokens = model.embed_tokens
    token_mask = token_mask.to(input_tokens.weight.device)

    pruned_tokens = input_tokens.weight * token_mask.unsqueeze(-1)

    with torch.no_grad():
        model_output = model.forward(input_tokens=pruned_tokens)

    head_mask = token_mask
    neuron_mask = token_mask
    
    # 计算延迟
    latency = estimate_latency(mha_lut, ffn_lut, head_mask, neuron_mask)
    
    return model_output, latency
