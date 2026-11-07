"""Numerical helpers retained at their original import path for compatibility.

Only the Glass-Box dependency closure is included, not the Jacobian fitter.
"""
import torch


def sync(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def unembed(model, state):
    """LFM3's final normalization is embedding_norm, not norm."""
    return model.lm_head(model.model.embedding_norm(state))
