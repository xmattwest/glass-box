"""Minimal loading helper, not a claim of support for arbitrary architectures.

The study adapter passes an explicit pinned revision. Calling this low-level
loader directly downloads weights and executes checkpoint code; its caller must
hold the inference lock for the entire load/use/unload lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

MODELS = {
    "instruct": "LiquidAI/LFM3-1.7B",
    "base": "LiquidAI/LFM3-1.7B-Base",
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Loaded:
    model: torch.nn.Module
    tokenizer: object
    repo_id: str
    revision: str
    device: str


def load(
    which: str = "instruct",
    revision: str | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> Loaded:
    """Load a checkpoint. Direct calls without a revision resolve the current head.

    Use ResearchSession.local for pinned identity and serialized ownership.
    """
    # Fixtures need Loaded, not Transformers or a Hugging Face connection.
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo_id = MODELS.get(which, which)
    revision = revision or model_info(repo_id).sha
    device = device or pick_device()

    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()
    return Loaded(model, tokenizer, repo_id, revision, device)
