# The decoder-order adaptation in GlassboxHooks.mid_forward is derived from
# Lfm3DecoderLayer.forward in the pinned modeling_lfm3.py:
# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
# Modified to intervene on the actual midpoint residual and restore on exit.
# See NOTICE and docs/provenance.md for the pinned source and scope.
"""Standalone LFM3 mechanistic primitives. No router dependencies or presets.

All positions refer to predictor tokens, not the tokens being predicted. Public
functions accept a Loaded model; its owner must hold inference_lock for the whole
load/use/unload lifetime and serialize hooks. DLA is a frozen-final-norm direct
 projection, not a downstream causal effect. See docs/library-guide.md.
"""
from __future__ import annotations

import fcntl
import gc
import json
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from lfm3lab.jacobian_lens import sync, unembed

REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
REPO_ID = "LiquidAI/LFM3-1.7B"
LOCK_PATH = Path(os.environ.get("LFM3_INFERENCE_LOCK",
    str(Path(tempfile.gettempdir()) / "opencode" / "lfm3-inference.lock")))
LAYER_TYPES = tuple("global" if i in (2, 6, 10, 14, 18, 22) else
                    "swa" if i in (4, 8, 12, 16, 20) else "conv" for i in range(24))
SCOPES = {"boundary", "first4", "answer", "prompt", "all"}


@contextmanager
def inference_lock(path=LOCK_PATH):
    """Hold until weights, graphs and caches have been released, not just forward."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def release_memory(device):
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class Unit:
    layer: int
    family: str
    index: int | str

    @classmethod
    def parse(cls, text, config=None):
        match = re.fullmatch(r"L(\d+)\.(?:([hcn])(\d+)|res\.(in|mid|out))", str(text))
        if not match:
            raise ValueError(f"Invalid canonical unit: {text!r}")
        layer, family, index, boundary = match.groups()
        unit = cls(int(layer), family or "res", boundary if boundary else int(index))
        types = config.layer_types if config else LAYER_TYPES
        if not 0 <= unit.layer < len(types):
            raise ValueError(f"Layer out of range: {text}")
        if unit.family == "res":
            return unit
        bounds = {"h": getattr(config, "num_attention_heads", 16),
                  "c": getattr(config, "hidden_size", 2048),
                  "n": getattr(config, "intermediate_size", 7680)}
        if not 0 <= unit.index < bounds[unit.family]:
            raise ValueError(f"Unit index out of range: {text}")
        if (unit.family == "h" and types[unit.layer] == "conv") or (
                unit.family == "c" and types[unit.layer] != "conv"):
            raise ValueError(f"Wrong unit family for layer: {text}")
        return unit

    @property
    def address(self):
        return f"L{self.layer}.{'res.' if self.family == 'res' else self.family}{self.index}"

    @property
    def checkpoint(self):
        if self.family != "res":
            raise ValueError("Only residual units have checkpoints")
        return f"out:{self.layer - 1}" if self.index == "in" and self.layer else f"{self.index}:{self.layer}"


@dataclass
class PatchSpec:
    """Ordered rules. Later rules win on overlapping coordinates.

    {"rules": [{"unit":"L2.h0", "op":"gate_clamp", "value":0,
                "scope":"boundary"}], "norm_mode":"native"}

    Ops: heads gate_clamp/logit_offset/scale/zero; conv clamp/scale/zero;
    MLP clamp/scale/zero/mean_scale (requires explicit calibrated mu);
    residual replace (source='clean', or explicit value vector), scale/zero.
    Optional positions narrows scope to absolute positions. row is reserved for
    batched sweeps. Residual donor sequences must have identical token counts.
    """
    rules: list[dict] = field(default_factory=list)
    norm_mode: str = "native"

    @classmethod
    def from_value(cls, value=None, config=None):
        if isinstance(value, cls):
            value = asdict(value)
        value = {} if value is None else value
        if not isinstance(value, dict) or set(value) - {"rules", "norm_mode"}:
            raise ValueError("PatchSpec requires only rules and norm_mode")
        spec = cls(**value)
        spec.validate(config)
        return spec

    def validate(self, config=None):
        if self.norm_mode not in ("native", "frozen") or not isinstance(self.rules, list):
            raise ValueError("Invalid norm_mode or rules")
        if len(self.rules) > 4096:
            raise ValueError("At most 4096 rules per PatchSpec")
        allowed = {"unit", "op", "value", "mu", "scope", "positions", "source", "row"}
        ops = {"h": {"gate_clamp", "logit_offset", "scale", "zero"},
               "c": {"clamp", "scale", "zero"}, "n": {"clamp", "scale", "zero", "mean_scale"},
               "res": {"replace", "scale", "zero"}}
        for rule in self.rules:
            if not isinstance(rule, dict) or set(rule) - allowed:
                raise ValueError("Unknown PatchSpec rule fields")
            unit = Unit.parse(rule.get("unit"), config)
            op = rule.get("op")
            if op not in ops[unit.family] or rule.get("scope", "boundary") not in SCOPES:
                raise ValueError(f"Invalid operation or scope for {unit.address}")
            if "source" in rule and (op != "replace" or rule["source"] != "clean"):
                raise ValueError("source is only valid for replace with source='clean'")
            if "row" in rule and (type(rule["row"]) is not int or rule["row"] < 0):
                raise ValueError("row must be a nonnegative integer")
            if "positions" in rule and (not isinstance(rule["positions"], list) or
                    any(type(p) is not int or p < 0 for p in rule["positions"])):
                raise ValueError("positions must be nonnegative integer indices")
            if op == "replace" and rule.get("source") == "clean":
                continue
            if op != "zero":
                value = rule.get("value")
                values = value if isinstance(value, list) and op == "replace" else [value]
                if not values or any(not isinstance(v, (float, int)) or isinstance(v, bool) or
                                     not math.isfinite(v) for v in values):
                    raise ValueError("A finite numeric value is required")
                if op == "gate_clamp" and not 0 <= value <= 1:
                    raise ValueError("gate_clamp must be in [0, 1]")
                if op == "replace" and (not isinstance(value, list) or len(value) != getattr(config, "hidden_size", 2048)):
                    raise ValueError("Residual replacement requires a hidden-size vector")
            if op == "mean_scale" and (not isinstance(rule.get("mu"), (float, int)) or
                                       not math.isfinite(rule["mu"])):
                raise ValueError("mean_scale needs an explicit finite mu; no hidden calibration")
        return self


def position_mask(positions, prompt_length, scope):
    relative = positions - (prompt_length - 1)
    if scope == "all":
        return torch.ones_like(positions, dtype=torch.bool)
    if scope == "boundary":
        return relative == 0
    if scope == "first4":
        return (relative >= 0) & (relative < 4)
    if scope == "answer":
        return relative >= 0
    if scope == "prompt":
        return relative < 0
    raise ValueError(scope)


def edit_values(values, rule, gate=False):
    op, value = rule["op"], rule.get("value", 0)
    if op == "gate_clamp":
        # Exact sigmoid endpoints. No +/-30 approximation.
        z = -math.inf if value == 0 else math.inf if value == 1 else math.log(value / (1 - value))
        return torch.full_like(values, z)
    if op == "logit_offset":
        return values + value
    if op in ("clamp", "replace"):
        return torch.zeros_like(values) + torch.as_tensor(value, device=values.device, dtype=values.dtype)
    if op == "zero":
        return torch.zeros_like(values)
    if op == "mean_scale":
        return rule["mu"] + value * (values - rule["mu"])
    if op == "scale":
        return values * value
    raise ValueError(op)


class GlassboxHooks:
    """Exception-safe intervention/capture hooks, with absolute cached cursors.

    Graph tensors stay on-device only for the lifetime of this context. A caller
    doing cached decoding uses capture=False, so old full activations are not kept.
    Norm and donor references are dictionaries of tensors from a matched clean pass.
    """
    def __init__(self, model, prompt_length, patch=None, *, reference=None, capture=True):
        self.model, self.prompt_length = model, prompt_length
        self.spec = PatchSpec.from_value(patch, model.config)
        self.reference = reference or {}
        self.capture = capture
        self.handles, self.cursors, self.forwards = [], {}, []
        self.acts, self.gates, self.points, self.inv, self.parts, self.routing = {}, {}, {}, {}, {}, {}
        self.rules = {}
        for rule in self.spec.rules:
            unit = Unit.parse(rule["unit"], model.config)
            key = unit.checkpoint if unit.family == "res" else (unit.family, unit.layer)
            self.rules.setdefault(key, []).append((unit, rule))

    def add(self, module, callback, pre=False):
        self.handles.append((module.register_forward_pre_hook if pre else module.register_forward_hook)(callback))

    def positions(self, key, x):
        start = self.cursors.get(key, 0)
        self.cursors[key] = start + x.shape[1]
        return torch.arange(start, start + x.shape[1], device=x.device)

    def apply(self, x, key, positions, gate=False):
        rules = [(u, r) for u, r in self.rules.get(key, []) if
                 (r["op"] in ("gate_clamp", "logit_offset")) == gate]
        if not rules:
            return x
        y = x.clone()
        for unit, rule in rules:
            mask = position_mask(positions, self.prompt_length, rule.get("scope", "boundary"))
            if "positions" in rule:
                mask &= torch.isin(positions, torch.tensor(rule["positions"], device=x.device))
            # Broadcast masks instead of nonzero/index_put (also works on MPS).
            rowmask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
            if "row" in rule:
                if rule["row"] >= x.shape[0]:
                    raise ValueError("Patch row outside batch")
                rowmask = torch.arange(x.shape[0], device=x.device) == rule["row"]
            mask = rowmask[:, None] & mask[None, :]
            if unit.family == "res":
                idx = slice(None)
            elif unit.family == "h" and not gate:
                d = self.model.config.head_dim
                idx = slice(unit.index * d, (unit.index + 1) * d)
            else:
                idx = unit.index
            vals = y[..., idx]
            if rule["op"] == "replace" and rule.get("source") == "clean":
                if key not in self.reference.get("points", {}):
                    raise ValueError(f"Missing clean residual donor for {key}")
                donor = self.reference["points"][key]
                if int(positions[-1]) >= donor.shape[1]:
                    raise ValueError("Residual donor is shorter than receiver")
                changed = donor[:, positions].to(vals)
            else:
                changed = edit_values(vals, rule, gate)
            y[..., idx] = torch.where(mask[..., None] if vals.ndim == 3 else mask, changed, vals)
        return y

    def projection(self, family, layer):
        key = (family, layer)
        def hook(m, args):
            x = self.apply(args[0], key, self.positions(("proj", key), args[0]))
            if self.capture:
                self.acts[key] = x
            return (x, *args[1:])
        return hook

    def residual(self, key, pre=False):
        def process(x):
            x = self.apply(x, key, self.positions(key, x))
            if self.capture:
                self.points[key] = x
            return x
        if pre:
            def hook(m, args):
                return (process(args[0]), *args[1:])
        else:
            def hook(m, args, output):
                return process(output)
        return hook

    def norm(self, layer, kind):
        key = (kind, layer)
        def hook(m, args, output):
            x = args[0]
            positions = self.positions(("norm", key), x)
            inv = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + m.variance_epsilon)
            if self.spec.norm_mode == "frozen":
                reference = self.reference.get("inv", {}).get(key)
                if reference is None:
                    raise ValueError("Frozen norms require a clean pass on the same prefix")
                # Cached clean hooks can expose just this step's denominator.
                inv = reference if reference.shape[1] == x.shape[1] else reference[:, positions]
                # LFM3 casts normalized states BEFORE multiplying the learned
                # weight. Reordering these BF16 roundings changes an identity run.
                output = m.weight * (x.float() * inv).to(x.dtype)
            self.inv[key] = inv.detach()
            return output
        return hook

    def mid_forward(self, block, layer):
        """An ffn_norm pre-hook alone edits only the MLP read, NOT its skip path.

        For a mid-residual intervention, reproduce the pinned decoder's forward
        and replace the actual residual between the two additions. Native
        submodules/hooks and operation order are unchanged. Restore on exit.
        """
        key = f"mid:{layer}"
        def forward(hidden_states, position_embeddings=None, attention_mask=None,
                    position_ids=None, past_key_values=None, seq_idx=None, **kwargs):
            residual = hidden_states
            if self.model.config.layer_types[layer] != "conv":
                hidden_states, _ = block.self_attn(
                    hidden_states=block.operator_norm(hidden_states),
                    position_embeddings=position_embeddings, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values, **kwargs)
            else:
                hidden_states = block.conv(hidden_states=block.operator_norm(hidden_states),
                    past_key_values=past_key_values, attention_mask=attention_mask, seq_idx=seq_idx)
            hidden_states = block.post_operator_norm(hidden_states) + residual
            hidden_states = self.apply(hidden_states, key, self.positions(key, hidden_states))
            if self.capture:
                self.points[key] = hidden_states
            ffn_out = block.post_ffn_norm(block.feed_forward(block.ffn_norm(hidden_states)))
            return hidden_states + ffn_out
        return forward

    def __enter__(self):
        try:
            for layer, block in enumerate(self.model.model.layers):
                if layer == 0:
                    self.add(block, self.residual("in:0", True), True)
                if f"mid:{layer}" in self.rules:
                    self.forwards.append((block, block.forward, "forward" in block.__dict__))
                    block.forward = self.mid_forward(block, layer)
                else:
                    self.add(block.ffn_norm, self.residual(f"mid:{layer}", True), True)
                self.add(block, self.residual(f"out:{layer}"))
                self.add(block.post_operator_norm, self.norm(layer, "op"))
                self.add(block.post_ffn_norm, self.norm(layer, "mlp"))
                self.add(block.feed_forward.w2, self.projection("n", layer), True)
                if self.model.config.layer_types[layer] == "conv":
                    self.add(block.conv.out_proj, self.projection("c", layer), True)
                    if self.capture:
                        def parts(m, a, o, l=layer):
                            self.parts[l] = o.chunk(3, -1)
                        self.add(block.conv.in_proj, parts)
                else:
                    self.add(block.self_attn.o_proj, self.projection("h", layer), True)
                    def gate(m, a, o, l=layer):
                        x = self.apply(o, ("h", l), self.positions(("gate", l), o), gate=True)
                        if self.capture:
                            self.gates[l] = x.sigmoid()
                        return x
                    self.add(block.self_attn.gate_proj, gate)
                    if self.capture:
                        def route(m, a, o, l=layer):
                            if o[1] is not None:
                                self.routing[l] = o[1]
                        self.add(block.self_attn, route)
            return self
        except BaseException:
            self.__exit__()
            raise

    def reference_copy(self):
        return {"inv": {k: v.detach().clone() for k, v in self.inv.items()},
                "points": {k: v.detach().clone() for k, v in self.points.items()}}

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for block, original, instance_override in self.forwards:
            if instance_override:
                block.forward = original
            else:
                del block.forward
        self.forwards.clear()
        for values in (self.acts, self.gates, self.points, self.inv, self.parts, self.routing):
            values.clear()


def architecture(model):
    c = model.config
    types = ["conv" if t == "conv" else "global" if t == "full_attention" else "swa" for t in c.layer_types]
    return {"layers": [{"layer": l, "type": t} for l, t in enumerate(types)],
            "heads": sum(t != "conv" for t in types) * c.num_attention_heads,
            "conv_channels": types.count("conv") * c.hidden_size,
            "mlp_neurons": len(types) * c.intermediate_size,
            "unique_residual_checkpoints": 2 * len(types) + 1,
            "conv_kernel_sizes": {str(l): int(b.conv.conv.weight.shape[-1]) for l, b in enumerate(model.model.layers)
                                  if types[l] == "conv"}}


def prepare_ids(lm, prompt, *, chat=True, thinking=False, max_tokens=512):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty prompt is required")
    text = lm.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking) if chat else prompt
    ids = lm.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(lm.device)
    if not 1 <= ids.shape[1] <= max_tokens:
        raise ValueError(f"Prompt has {ids.shape[1]} tokens; budget is {max_tokens}")
    return ids


def explicit_ids(lm, value, max_tokens=768):
    """Validate JSON/list or tensor token IDs, with no arbitrary tensor uploads."""
    if isinstance(value, list):
        if not value or any(type(v) is not int for v in value):
            raise ValueError("input_ids must be a flat nonempty integer list")
        value = torch.tensor([value], dtype=torch.long, device=lm.device)
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1:
        raise ValueError("input_ids must contain exactly one sequence")
    if not 1 <= value.shape[1] <= max_tokens:
        raise ValueError(f"Explicit input token budget: 1..{max_tokens}")
    if value.dtype != torch.long or int(value.min()) < 0 or int(value.max()) >= lm.model.lm_head.weight.shape[0]:
        raise ValueError("input_ids outside vocabulary or not int64")
    return value.to(lm.device)


def resolve_pair(tokenizer, logits, target=None, foil=None):
    def one(value):
        if type(value) is int:
            tid = value
        elif isinstance(value, str) and value:
            ids = tokenizer.encode(value, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Target/foil must each be one token or an integer token ID: {value!r} -> {ids}")
            tid = ids[0]
        else:
            raise ValueError("Invalid target/foil")
        if not 0 <= tid < logits.shape[-1]:
            raise ValueError("Token ID out of vocabulary")
        return tid
    top = logits.float().topk(2).indices.tolist()
    a = top[0] if target is None else one(target)
    b = next(i for i in top if i != a) if foil is None else one(foil)
    if a == b:
        raise ValueError("Target and foil must differ")
    return a, b


def token_summary(tok, logits, chosen=None):
    z = logits.float()
    lp = z.log_softmax(-1)
    values, indices = z.topk(min(5, z.shape[-1]))
    chosen = int(indices[0]) if chosen is None else int(chosen)
    return {"token_id": chosen, "token": tok.decode([chosen]), "prob": float(lp[chosen].exp()),
            "logprob": float(lp[chosen]), "margin": float(values[0] - values[1]),
            "top5": [{"id": int(i), "token": tok.decode([int(i)]), "prob": float(lp[i].exp())}
                     for i in indices]}


def _reference(model, ids, prompt_length, spec, donor_ids=None):
    need_points = any(r.get("source") == "clean" for r in spec.rules)
    if spec.norm_mode == "native" and not need_points:
        return None
    with torch.no_grad(), GlassboxHooks(model, prompt_length, capture=need_points) as clean:
        model.model(input_ids=ids, use_cache=False)
        reference = clean.reference_copy()
    if donor_ids is not None and need_points:
        if donor_ids.shape != ids.shape:
            raise ValueError("Clean/corrupt donor and receiver must have the same token count")
        with torch.no_grad(), GlassboxHooks(model, prompt_length) as donor:
            model.model(input_ids=donor_ids, use_cache=False)
            reference["points"] = donor.reference_copy()["points"]
    return reference


def _at_position(position, length):
    position = length - 1 if position is None else position
    if type(position) is not int or not 0 <= position < length:
        raise ValueError("position must be an absolute input-token index")
    return position


def _lag_terms(model, cap, layer, index, position):
    b, c, x = cap.parts[layer]
    kernel = model.model.layers[layer].conv.conv.weight[index, 0].float().flip(0)
    terms = [float(c[0, position, index].float() * kernel[lag] *
                   b[0, position - lag, index].float() * x[0, position - lag, index].float())
             if position >= lag else 0.0 for lag in range(len(kernel))]
    conv = model.model.layers[layer].conv.conv
    bias = 0.0 if conv.bias is None else float(c[0, position, index].float() * conv.bias[index].float())
    activation = float(cap.acts[("c", layer)][0, position, index])
    return {"terms": terms, "bias": bias, "activation": activation,
            "B": float(b[0, position, index]), "C": float(c[0, position, index]), "x": float(x[0, position, index]),
            "reconstruction_error": abs(sum(terms) + bias - activation),
            "note": "pre-edit kernel decomposition; edits and dtype rounding may leave a residual"}


@contextmanager
def attention_mode(model, eager=False):
    old = model.config._attn_implementation
    if eager:
        model.config._attn_implementation = "eager"
    try:
        yield
    finally:
        model.config._attn_implementation = old


@torch.no_grad()
def run_glassbox_probe(lm, prompt=None, *, patch=None, target=None, foil=None,
                       position=None, top_k=8, pins=None, routing=False, chat=True,
                       thinking=False, input_ids=None, prompt_length=None, donor_prompt=None,
                       trace_start=None, chosen_ids=None, vocabulary_units=None):
    """One instrumented forward (+ clean reference only when required).

    Routing is opt-in, limited to 128 tokens and 8 pinned heads, to avoid a huge
    T*T JSON payload. Unit rankings are direct projections, NOT proven causality.
    Logit lens is evaluated at the inspected position across every layer.
    """
    model, tok = lm.model, lm.tokenizer
    ids = explicit_ids(lm, input_ids) if input_ids is not None else prepare_ids(lm, prompt, chat=chat, thinking=thinking)
    if ids.shape[0] != 1 or ids.shape[1] > 768:
        raise ValueError("Probe budget: one sequence, at most 768 tokens")
    length = ids.shape[1]
    p = _at_position(position, length)
    prompt_length = prompt_length or length
    if type(prompt_length) is not int or not 1 <= prompt_length <= length:
        raise ValueError("prompt_length must be 1..sequence length")
    top_k = int(top_k)
    if not 1 <= top_k <= 64:
        raise ValueError("top_k must be 1..64")
    pins = pins or []
    if len(pins) > 32:
        raise ValueError("At most 32 pinned units")
    pin_units = [Unit.parse(s, model.config) for s in pins]
    vocabulary_units = vocabulary_units or []
    if len(vocabulary_units) > 4:
        raise ValueError("At most four vocabulary projections per probe")
    route_heads = [u for u in pin_units if u.family == "h"]
    if routing and (length > 128 or not 1 <= len(route_heads) <= 8):
        raise ValueError("Routing requires 1..8 pinned heads and at most 128 tokens")
    spec = PatchSpec.from_value(patch, model.config)
    donor_ids = prepare_ids(lm, donor_prompt, chat=chat, thinking=thinking) if donor_prompt else None
    sync(lm.device)
    start = time.perf_counter()
    reference = _reference(model, ids, prompt_length, spec, donor_ids)
    with attention_mode(model, routing), GlassboxHooks(model, prompt_length, spec, reference=reference) as cap:
        model.model(input_ids=ids, use_cache=False)
        last = len(model.model.layers) - 1
        final = cap.points[f"out:{last}"]
        logits = unembed(model, final[:, p])[0]
        a, b = resolve_pair(tok, logits, target, foil)
        norm = model.model.embedding_norm
        final_inv = torch.rsqrt(final[0].float().square().mean(-1, keepdim=True) + norm.variance_epsilon)
        direction = (model.lm_head.weight[a].float() - model.lm_head.weight[b].float()) * norm.weight.float()
        d = direction * final_inv[p]
        attn_layers = [l for l, t in enumerate(model.config.layer_types) if t != "conv"]
        matrices = {k: [] for k in ("gate", "write_energy", "dla", "temporal_sd")}
        timeline = {k: [] for k in ("gate", "write_energy", "dla")}
        for l in attn_layers:
            block = model.model.layers[l]
            g = cap.gates[l][0].float()
            z = cap.acts[("h", l)][0].float().view(length, model.config.num_attention_heads, -1)
            w = block.self_attn.o_proj.weight.float() * block.post_operator_norm.weight.float()[:, None]
            w = w.view(w.shape[0], model.config.num_attention_heads, -1)
            # Grams amortize exact per-head residual-write energies across tokens.
            grams = getattr(model, "_glassbox_grams", {})
            if l not in grams:
                grams[l] = torch.einsum("dhk,dhj->hkj", w, w)
                model._glassbox_grams = grams
            inv = cap.inv[("op", l)][0]
            energy = torch.einsum("thk,hkj,thj->th", z, grams[l], z).clamp_min(0).sqrt() * inv
            projected = torch.einsum("d,dhk->hk", direction, w)
            dla = (z * projected).sum(-1) * inv * final_inv
            for key, value in (("gate", g), ("write_energy", energy), ("dla", dla)):
                matrices[key].append(value[p].cpu().tolist())
                timeline[key].append(value.cpu().tolist())
            matrices["temporal_sd"].append(g.std(0, correction=0).cpu().tolist())
        states = torch.stack([cap.points[f"out:{l}"][0, p] for l in range(last + 1)])
        lens = unembed(model, states).float()
        probs = lens.softmax(-1)
        spine, units = [], {}
        for l, block in enumerate(model.model.layers):
            before = cap.points["in:0" if l == 0 else f"out:{l-1}"][0, p].float()
            mid, after = cap.points[f"mid:{l}"][0, p].float(), cap.points[f"out:{l}"][0, p].float()
            op, mlp = mid - before, after - mid
            kind = "conv" if model.config.layer_types[l] == "conv" else "global" if model.config.layer_types[l] == "full_attention" else "swa"
            best = int(lens[l].argmax())
            spine.append({"layer": l, "type": kind, "top_id": best, "top_token": tok.decode([best]),
                          "target_prob": float(probs[l, a]), "foil_prob": float(probs[l, b]),
                          "lens_margin": float(lens[l, a] - lens[l, b]),
                          "operator_norm": float(op.norm()), "mlp_norm": float(mlp.norm()),
                          "operator_dla": float(op @ d), "mlp_dla": float(mlp @ d)})
            entries = {}
            for family in (["n", "c"] if kind == "conv" else ["n"]):
                proj = block.feed_forward.w2 if family == "n" else block.conv.out_proj
                post = block.post_ffn_norm if family == "n" else block.post_operator_norm
                inv = cap.inv[("mlp" if family == "n" else "op", l)][0, p]
                act = cap.acts[(family, l)][0, p].float()
                scores = act * ((d * post.weight.float()) @ proj.weight.float()) * inv
                indices = scores.abs().topk(min(top_k, len(scores))).indices.tolist()
                entries[family] = []
                for n in indices:
                    write = proj.weight[:, n].float() * post.weight.float() * inv * act[n]
                    entry = {"unit": f"L{l}.{family}{n}", "activation": float(act[n]),
                             "dla": float(scores[n]), "write_energy": float(write.norm())}
                    if family == "c":
                        entry["lags"] = _lag_terms(model, cap, l, n, p)
                    entries[family].append(entry)
            units[str(l)] = entries
        temporal = {}
        for u in pin_units:
            if u.family == "res":
                values = cap.points[u.checkpoint][0].float().norm(dim=-1)
                measure = "residual_l2_norm"
            elif u.family == "h":
                values, measure = cap.gates[u.layer][0, :, u.index], "gate_sigma"
            else:
                values, measure = cap.acts[(u.family, u.layer)][0, :, u.index], "pre_projection_activation"
            temporal[u.address] = {"measure": measure, "values": values.float().cpu().tolist()}
        vocabulary = {}
        for address in vocabulary_units:
            u = Unit.parse(address, model.config)
            if u.family not in ("n", "c"):
                raise ValueError("Vocabulary projections require conv channels or MLP neurons")
            block = model.model.layers[u.layer]
            proj = block.feed_forward.w2 if u.family == "n" else block.conv.out_proj
            post = block.post_ffn_norm if u.family == "n" else block.post_operator_norm
            write = proj.weight[:, u.index].float() * post.weight.float() * cap.inv[("mlp" if u.family == "n" else "op", u.layer)][0, p]
            write *= cap.acts[(u.family, u.layer)][0, p, u.index].float()
            # Chunked fp32 matvec avoids a full fp32 copy of the 128k x 2048 head.
            read = write * norm.weight.float() * final_inv[p]
            scores = torch.cat([weight.float() @ read for weight in model.lm_head.weight.split(4096)])
            def words(sign):
                _, ns = (scores * sign).topk(top_k)
                return [{"id": int(n), "token": tok.decode([int(n)]), "direct_logit": float(scores[n])} for n in ns]
            vocabulary[address] = {"positive": words(1), "negative": words(-1),
                                   "definition": "activated unit write projected with frozen post and final RMS denominators"}
        trace_start = 0 if trace_start is None else trace_start
        traces = []
        # Chunk vocabulary readouts: never retain a 768 x vocabulary GPU tensor.
        for begin in range(trace_start, length, 16):
            zs = unembed(model, final[0, begin:min(length, begin + 16)])
            for j, z in enumerate(zs):
                pos = begin + j
                chosen = chosen_ids[pos - trace_start] if chosen_ids is not None else None
                traces.append({"position": pos, **token_summary(tok, z, chosen)})
        routes = {u.address: cap.routing[u.layer][0, u.index].float().cpu().tolist() for u in route_heads} if routing else {}
        final_identity_error = float((lens[-1] - logits.float()).abs().max())
        result = {"kind": "probe", "repo_id": lm.repo_id, "revision": lm.revision, "device": str(lm.device),
                  "dtype": str(next(model.parameters()).dtype), "patch": asdict(spec), "architecture": architecture(model),
                  "input_ids": ids[0].tolist(), "tokens": [tok.decode([i]) for i in ids[0].tolist()],
                  "prompt_length": prompt_length, "position": p, "target": {"id": a, "text": tok.decode([a])},
                  "foil": {"id": b, "text": tok.decode([b])}, "margin": float(logits[a].float() - logits[b].float()),
                  "decision": token_summary(tok, logits), "head_layers": attn_layers, "matrices": matrices,
                  "head_timeline": timeline, "spine": spine, "units": units, "temporal": temporal,
                  "token_trace": traces, "routing": routes, "vocabulary": vocabulary,
                  "final_lens_identity_error": final_identity_error,
                  "dla": {"definition": "post-sandwich write dot (W_A-W_B) * final RMS weight / final-state RMS",
                          "causal": False, "unit_ranking": "absolute direct projection, not causal ranking",
                          "rounding_note": "unit decomposition uses fp32 projection; native low-precision norm/residual roundings are not distributed among units",
                          "embedding_contribution": float(cap.points["in:0"][0, p].float() @ d),
                          "spine_sum": float((final[0, p].float() - cap.points["in:0"][0, p].float()) @ d)}}
    del reference
    sync(lm.device)
    result["elapsed_s"] = time.perf_counter() - start
    return result


def run_gradient_attribution(lm, prompt=None, *, patch=None, target=None, foil=None, position=None,
                             top_k=8, chat=True, thinking=False, input_ids=None,
                             prompt_length=None, full_scores=False):
    """One differentiable forward + one autograd.grad, no weight gradients.

    Head score: sum_d z_d*dmargin/dz_d, with z=g*attention, algebraically
    g*dmargin/dg. Rankings sum over all causal source positions up to position.
    Inspected-position and per-position head arrays are also returned.
    """
    model, tok = lm.model, lm.tokenizer
    ids = explicit_ids(lm, input_ids, 256) if input_ids is not None else prepare_ids(lm, prompt, chat=chat, thinking=thinking, max_tokens=256)
    if ids.shape[0] != 1 or ids.shape[1] > 256 or not 1 <= int(top_k) <= 64:
        raise ValueError("Gradient budget: one sequence, <=256 tokens, top_k 1..64")
    p = _at_position(position, ids.shape[1])
    prompt_length = prompt_length or ids.shape[1]
    if type(prompt_length) is not int or not 1 <= prompt_length <= ids.shape[1]:
        raise ValueError("Invalid prompt_length")
    spec = PatchSpec.from_value(patch, model.config)
    sync(lm.device)
    start = time.perf_counter()
    reference = _reference(model, ids, prompt_length, spec)
    # Starting graph at embeddings avoids allocating 1.7B parameter gradients.
    flags = [v.requires_grad for v in model.parameters()]
    model.requires_grad_(False)
    try:
        with torch.enable_grad(), GlassboxHooks(model, prompt_length, spec, reference=reference) as cap:
            embeds = model.model.embed_tokens(ids).detach().requires_grad_(True)
            model.model(inputs_embeds=embeds, use_cache=False)
            z = unembed(model, cap.points[f"out:{len(model.model.layers)-1}"][:, p])[0].float()
            a, b = resolve_pair(tok, z, target, foil)
            objective = z[a] - z[b]
            keys = list(cap.acts)
            sources = [cap.acts[k] for k in keys]
            gradients = torch.autograd.grad(objective, sources)
            heads, head_at, head_time, rankings, arrays = {}, {}, {}, {}, {}
            for key, act, grad in zip(keys, sources, gradients, strict=True):
                family, layer = key
                scores = (act.detach().float() * grad.detach().float())[0]
                if family == "h":
                    scores = scores.view(ids.shape[1], model.config.num_attention_heads, -1).sum(-1)
                    heads[layer] = scores[:p+1].sum(0).cpu().tolist()
                    head_at[layer] = scores[p].cpu().tolist()
                    head_time[layer] = scores.cpu().tolist()
                reduced = scores[:p+1].sum(0)
                indices = reduced.abs().topk(min(int(top_k), len(reduced))).indices.tolist()
                rankings[f"{family}:{layer}"] = [{"unit": f"L{layer}.{family}{i}", "score": float(reduced[i]),
                                                  "at_position": float(scores[p, i])} for i in indices]
                if full_scores:
                    arrays[f"{family}:{layer}"] = reduced.cpu().tolist()
            head_layers = sorted(heads)
            result = {"kind": "attribute", "revision": lm.revision, "device": str(lm.device),
                      "patch": asdict(spec), "target": {"id": a, "text": tok.decode([a])},
                      "foil": {"id": b, "text": tok.decode([b])}, "position": p,
                      "input_ids": ids[0].tolist(), "prompt_length": prompt_length,
                      "margin": float(objective.detach()), "head_layers": head_layers,
                      "head_scores": [heads[l] for l in head_layers], "head_at_position": [head_at[l] for l in head_layers],
                      "head_timeline": [head_time[l] for l in head_layers], "rankings": rankings,
                      "scores": arrays, "architecture": architecture(model),
                      "definition": "activation * gradient of target-minus-foil logit; sum over causal source positions",
                      "head_definition": "gated pre-o_proj vector dot gradient, equivalent to gate * dmargin/dgate",
                      "forward_passes": 1, "backward_passes": 1,
                      "reference_passes": int(reference is not None)}
        del gradients, sources, embeds, objective, z, act, grad
    finally:
        for parameter, flag in zip(model.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)
    del reference
    sync(lm.device)
    result["elapsed_s"] = time.perf_counter() - start
    return result


@torch.no_grad()
def run_head_ablation_sweep(lm, prompt, *, target=None, foil=None, scope="boundary", norm_mode="native",
                            batch_size=4, units=None, mode="heads", chat=True, thinking=False):
    """Batched exact leave-one-out effects (patched minus clean logit margin).

    mode=layers removes each operator write; mode=mlp_layers each MLP write;
    mode=blocks bypasses the entire decoder block. Clean comparison uses the
    same batch shape as its intervention, avoiding batch-kernel baseline drift.
    """
    model, tok = lm.model, lm.tokenizer
    ids = prepare_ids(lm, prompt, chat=chat, thinking=thinking, max_tokens=256)
    if not 1 <= batch_size <= 16 or mode not in ("heads", "layers", "mlp_layers", "blocks") or scope not in SCOPES:
        raise ValueError("Invalid sweep mode, scope or batch size (1..16)")
    if str(lm.device).startswith("mps") and batch_size * ids.shape[1] > 1024:
        raise ValueError("MPS sweep budget: batch_size * prompt_tokens <= 1024")
    PatchSpec(norm_mode=norm_mode).validate(model.config)
    if mode == "heads":
        units = units if units is not None else [f"L{l}.h{h}" for l, t in enumerate(model.config.layer_types)
                    if t != "conv" for h in range(model.config.num_attention_heads)]
        parsed = [Unit.parse(u, model.config) for u in units]
        if not parsed or any(u.family != "h" for u in parsed) or len(set(units)) != len(units):
            raise ValueError("Head sweep requires unique head addresses")
    else:
        if units is not None:
            raise ValueError("units is only supported for head sweeps")
        label = {"mlp_layers": "mlp", "layers": "operator", "blocks": "block"}[mode]
        units = [f"L{l}.{label}" for l in range(len(model.model.layers))]
    start = time.perf_counter()
    with GlassboxHooks(model, ids.shape[1], capture=False) as clean:
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
        logits = model.lm_head(hidden)[0].float()
        a, b = resolve_pair(tok, logits, target, foil)
        baseline = float(logits[a] - logits[b])
        reference = clean.reference_copy()
    rows = []
    for start_row in range(0, len(units), batch_size):
        batch = units[start_row:start_row + batch_size]
        with GlassboxHooks(model, ids.shape[1], capture=False) as clean:
            hidden = model.model(input_ids=ids.expand(len(batch), -1), use_cache=False).last_hidden_state[:, -1]
            z = model.lm_head(hidden).float()
            clean_margins = (z[:, a] - z[:, b]).cpu().tolist()
            reference = clean.reference_copy()
        rules = []
        handles = []
        try:
            if mode == "heads":
                rules = [{"unit": u, "op": "zero", "scope": scope, "row": row} for row, u in enumerate(batch)]
            else:
                for row, label in enumerate(batch):
                    layer = int(label.split(".")[0][1:])
                    block = model.model.layers[layer]
                    norm = block if mode == "blocks" else block.post_ffn_norm if mode == "mlp_layers" else block.post_operator_norm
                    def knockout(m, args, out, r=row):
                        y = out.clone()
                        mask = position_mask(torch.arange(out.shape[1], device=out.device), ids.shape[1], scope)
                        y[r] = torch.where(mask[:, None], args[0][r] if mode == "blocks" else 0, y[r])
                        return y
                    # Install AFTER GlassboxHooks below, so frozen norms cannot overwrite knockout.
                    handles.append((norm, knockout))
            with GlassboxHooks(model, ids.shape[1], PatchSpec(rules, norm_mode), reference=reference, capture=False):
                registered = [m.register_forward_hook(h) for m, h in handles]
                try:
                    h = model.model(input_ids=ids.expand(len(batch), -1), use_cache=False).last_hidden_state[:, -1]
                    z = model.lm_head(h).float()
                    margins = (z[:, a] - z[:, b]).cpu().tolist()
                finally:
                    for handle in registered:
                        handle.remove()
            rows.extend({"unit": u, "margin": margin, "clean_margin": base, "delta_margin": margin - base}
                        for u, margin, base in zip(batch, margins, clean_margins, strict=True))
        finally:
            handles.clear()
    sync(lm.device)
    return {"kind": "sweep", "mode": mode, "revision": lm.revision, "device": str(lm.device),
            "prompt": prompt, "input_ids": ids[0].tolist(), "target": {"id": a, "text": tok.decode([a])},
            "foil": {"id": b, "text": tok.decode([b])}, "scope": scope, "norm_mode": norm_mode,
            "batch_size": batch_size, "baseline_margin": baseline, "rows": rows,
            "conditions": len(rows), "elapsed_s": time.perf_counter() - start,
            "definition": "patched minus same-batch-shape clean target-minus-foil logit; measured finite intervention in recorded dtype",
            "dtype": str(next(model.parameters()).dtype),
            "max_batch_baseline_drift": max(abs(r["clean_margin"] - baseline) for r in rows)}


@torch.no_grad()
def run_glassbox_generate(lm, prompt, *, patch=None, target=None, foil=None, max_new_tokens=32,
                          top_k=8, pins=None, chat=True, thinking=False):
    """Greedy cached generation, then full teacher-forced mechanistic replay.

    Frozen norms use a separate clean cache fed exactly the patched trajectory.
    Replay gives synchronized telemetry even after generated tokens diverge.
    Clean-vs-patched decision comparison should use fixed A/B IDs.
    """
    if not 1 <= max_new_tokens <= 128:
        raise ValueError("max_new_tokens must be 1..128")
    model, tok = lm.model, lm.tokenizer
    ids = prepare_ids(lm, prompt, chat=chat, thinking=thinking)
    n = ids.shape[1]
    spec = PatchSpec.from_value(patch, model.config)
    if any(r.get("source") == "clean" for r in spec.rules):
        raise ValueError("Residual donor replacement is a probe operation; generation requires explicit vector donors")
    reference = {"inv": {}}
    cache = clean_cache = None
    current = ids
    generated, trace = [], []
    start = time.perf_counter()
    # Alternate clean/patched hooks rather than simultaneously instrumenting one model.
    patched = GlassboxHooks(model, n, spec, reference=reference, capture=False)
    clean = GlassboxHooks(model, n, capture=False)
    for step in range(max_new_tokens):
        if spec.norm_mode == "frozen":
            with clean:
                out = model.model(input_ids=current, past_key_values=clean_cache, use_cache=True)
                clean_cache = out.past_key_values
                reference["inv"] = {k: v.clone() for k, v in clean.inv.items()}
        with patched:
            out = model.model(input_ids=current, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = model.lm_head(out.last_hidden_state[:, -1])[0]
        if step == 0:
            a, b = resolve_pair(tok, logits, target, foil)
        tid = int(logits.argmax())
        generated.append(tid)
        trace.append({"position": n - 1 + step, **token_summary(tok, logits, tid)})
        if tid == tok.eos_token_id:
            break
        current = torch.tensor([[tid]], device=ids.device)
    sync(lm.device)
    generation_seconds = time.perf_counter() - start
    replay = torch.cat([ids, torch.tensor([generated[:-1]], dtype=ids.dtype, device=ids.device)], -1)
    del cache, clean_cache, out, patched, clean, reference
    result = run_glassbox_probe(lm, input_ids=replay, prompt_length=n, patch=spec, target=a, foil=b,
                               top_k=top_k, pins=pins, trace_start=n - 1, chosen_ids=generated)
    replay_error = max(abs(x["logprob"] - y["logprob"]) for x, y in zip(trace, result["token_trace"], strict=True))
    result.update(kind="generate", generated_ids=generated, text=tok.decode(generated, skip_special_tokens=False),
                  cached_token_trace=trace, cached_replay_max_logprob_error=replay_error,
                  generation_s=generation_seconds, total_s=time.perf_counter() - start,
                  comparison_policy="teacher-force clean probe on input_ids for matched-token deltas")
    return result


# Stable short aliases for agent clients.
probe_pass = run_glassbox_probe
attribute_gradient = run_gradient_attribution
ablation_sweep = run_head_ablation_sweep


def synthetic_glassbox_checks():
    """Offline CPU tests with real modules/hooks, all 24 hybrid layers, no weights download."""
    from types import SimpleNamespace
    from torch import nn
    from torch.nn import functional as F
    from lfm3lab.model import Loaded

    class Norm(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = nn.Parameter(torch.linspace(0.8, 1.2, width))
            self.variance_epsilon = 1e-6

        def forward(self, x):
            return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.variance_epsilon)).to(x.dtype) * self.weight

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1, self.w3, self.w2 = nn.Linear(8, 12, False), nn.Linear(8, 12, False), nn.Linear(12, 8, False)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj, self.out_proj = nn.Linear(8, 24, False), nn.Linear(8, 8, False)
            self.conv = nn.Conv1d(8, 8, 4, padding=3, groups=8, bias=False)

        def forward(self, h=None, hidden_states=None, **kwargs):
            h = hidden_states if hidden_states is not None else h
            b, c, x = self.in_proj(h).chunk(3, -1)
            return self.out_proj(c * self.conv((b * x).transpose(1, 2))[..., :h.shape[1]].transpose(1, 2))

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.o_proj = nn.Linear(8, 2, False), nn.Linear(8, 8, False)
            self.v_proj = nn.Linear(8, 8, False)

        def forward(self, h=None, hidden_states=None, **kwargs):
            h = hidden_states if hidden_states is not None else h
            v = self.v_proj(h).cumsum(1) / torch.arange(1, h.shape[1] + 1)[None, :, None]
            gated = v.view(*h.shape[:2], 2, 4) * self.gate_proj(h).sigmoid()[..., None]
            return self.o_proj(gated.flatten(-2)), None

    class Block(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.operator_norm, self.ffn_norm = Norm(8), Norm(8)
            self.post_operator_norm, self.post_ffn_norm = Norm(8), Norm(8)
            self.feed_forward = MLP()
            if kind == "conv":
                self.conv = Conv()
            else:
                self.self_attn = Attn()

        def forward(self, h):
            x = self.operator_norm(h)
            h = h + self.post_operator_norm(self.conv(x) if self.kind == "conv" else self.self_attn(x)[0])
            return h + self.post_ffn_norm(self.feed_forward(self.ffn_norm(h)))

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(t) for t in LAYER_TYPES])
            self.embed_tokens, self.embedding_norm = nn.Embedding(32, 8), Norm(8)

        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            for block in self.layers:
                h = block(h)
            return SimpleNamespace(last_hidden_state=self.embedding_norm(h))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(layer_types=["full_attention" if t == "global" else
                       "sliding_attention" if t == "swa" else t for t in LAYER_TYPES],
                       num_attention_heads=2, head_dim=4, hidden_size=8, intermediate_size=12,
                       _attn_implementation="sdpa")
            self.model, self.lm_head = Body(), nn.Linear(8, 32, False)

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ":".join(map(str, ids))

    with torch.random.fork_rng():
        torch.manual_seed(24)
        model = Tiny().eval()
    lm = Loaded(model, Tokenizer(), "synthetic", "synthetic", "cpu")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    checks = {}
    with torch.no_grad(), GlassboxHooks(model, 6) as cap:
        model.model(ids)
        reference = cap.reference_copy()
        baseline = {k: v.clone() for k, v in cap.acts.items()}
        gates = {k: v.clone() for k, v in cap.gates.items()}
        expected = unembed(model, cap.points["out:23"][:, -1])
        assert len(cap.points) == 49
        errors = [_lag_terms(model, cap, l, n, p)["reconstruction_error"]
                  for l, t in enumerate(LAYER_TYPES) if t == "conv" for n in range(8) for p in range(6)]
        assert max(errors) < 1e-6
        checks["four_lag_reconstruction_max_error"] = max(errors)
    # Repeat exact reconstruction with the pinned checkpoint's actual width.
    with torch.random.fork_rng():
        torch.manual_seed(2407)
        original = model.model.layers[0].conv.conv
        model.model.layers[0].conv.conv = nn.Conv1d(8, 8, 7, padding=6, groups=8, bias=False)
        long_ids = torch.tensor([list(range(1, 10))])
        with torch.no_grad(), GlassboxHooks(model, 9) as cap:
            model.model(long_ids)
            errors = [_lag_terms(model, cap, 0, n, p)["reconstruction_error"] for n in range(8) for p in range(9)]
        model.model.layers[0].conv.conv = original
        assert max(errors) < 1e-6
        checks["seven_lag_reconstruction_max_error"] = max(errors)
    checkpoints = {Unit.parse(f"L{l}.res.{p}").checkpoint for l in range(24) for p in ("in", "mid", "out")}
    assert len(checkpoints) == 49
    checks["residual_aliases_72_to_49"] = True
    for invalid in ("L0.h0", "L2.c0", "L24.n0", "L0.n7680", "L2h0"):
        try:
            Unit.parse(invalid)
            raise AssertionError(f"Accepted {invalid}")
        except ValueError:
            pass
    for invalid in ({"rules": [{"unit": "L2.h0", "op": "gate_clamp", "value": 2}]},
                    {"rules": [{"unit": "L0.n0", "op": "mean_scale", "value": 2}]},
                    {"norm_mode": "unknown"}):
        try:
            PatchSpec.from_value(invalid)
            raise AssertionError("Accepted invalid patch")
        except ValueError:
            pass
    checks["validation"] = True
    # Direct production hooks across prefill + cached single-token calls.
    rules = [{"unit": "L2.h0", "op": "gate_clamp", "value": 0, "scope": "boundary"},
             {"unit": "L4.h1", "op": "logit_offset", "value": 2, "scope": "first4"},
             {"unit": "L0.n1", "op": "mean_scale", "value": 3, "mu": 0.25, "scope": "answer"},
             {"unit": "L0.c2", "op": "clamp", "value": 4, "scope": "prompt"}]
    with torch.no_grad(), GlassboxHooks(model, 3, PatchSpec(rules)) as cap:
        for step, width in enumerate((3, 1, 1, 1, 1)):
            h = torch.ones(1, width, 8)
            for layer in (2, 4):
                gate_proj = model.model.layers[layer].self_attn.gate_proj
                raw = F.linear(h, gate_proj.weight)
                edited = gate_proj(h)
                if layer == 2:
                    if step == 0:
                        assert float(edited[0, -1, 0].sigmoid()) == 0
                        torch.testing.assert_close(edited[0, :-1], raw[0, :-1])
                    else:
                        torch.testing.assert_close(edited, raw)
                elif step < 4:
                    torch.testing.assert_close(edited[0, -1, 1], raw[0, -1, 1] + 2)
                else:
                    torch.testing.assert_close(edited, raw)
            model.model.layers[0].feed_forward.w2(torch.ones(1, width, 12))
            got = cap.acts[("n", 0)][0, -1, 1]
            torch.testing.assert_close(got, torch.tensor(2.5))
            model.model.layers[0].conv.out_proj(torch.ones(1, width, 8))
            torch.testing.assert_close(cap.acts[("c", 0)][0, -1, 2], torch.tensor(1.0))
            if step == 0:
                torch.testing.assert_close(cap.acts[("c", 0)][0, :2, 2], torch.full((2,), 4.0))
    checks["scoped_gate_conv_mlp_cached_positions"] = True
    torch.testing.assert_close(edit_values(torch.zeros(2), {"op": "gate_clamp", "value": 1}).sigmoid(), torch.ones(2))
    checks["exact_gate_endpoints"] = True
    # Full no-op, scoped coordinate isolation, native vs clean denominator.
    with torch.no_grad(), GlassboxHooks(model, 6, PatchSpec(norm_mode="frozen"), reference=reference) as cap:
        model.model(ids)
        torch.testing.assert_close(unembed(model, cap.points["out:23"][:, -1]), expected, atol=0, rtol=0)
    rule = {"unit": "L0.c1", "op": "zero"}
    with torch.no_grad(), GlassboxHooks(model, 6, PatchSpec([rule], "frozen"), reference=reference) as cap:
        model.model(ids)
        torch.testing.assert_close(cap.inv[("op", 0)], reference["inv"][("op", 0)], atol=0, rtol=0)
        torch.testing.assert_close(cap.acts[("c", 0)][:, :-1], baseline[("c", 0)][:, :-1])
        assert cap.acts[("c", 0)][0, -1, 1] == 0
        frozen_state = cap.points["mid:0"].clone()
    with torch.no_grad(), GlassboxHooks(model, 6, PatchSpec([rule])) as cap:
        model.model(ids)
        assert not torch.allclose(cap.points["mid:0"], frozen_state)
    checks["frozen_identity_and_native_difference"] = True
    corrupt = ids.clone()
    corrupt[0, 1] = 7
    for address in ("L0.res.in", "L0.res.mid", "L0.res.out", "L1.res.in", "L2.res.mid"):
        with torch.no_grad(), GlassboxHooks(model, 6, PatchSpec([
                {"unit": address, "op": "replace", "source": "clean", "scope": "all"}]), reference=reference) as cap:
            model.model(corrupt)
            torch.testing.assert_close(unembed(model, cap.points["out:23"][:, -1]), expected, atol=0, rtol=0)
    checks["clean_corrupt_residual_transplant"] = True
    probe = run_glassbox_probe(lm, input_ids=ids, target=1, foil=2, top_k=2, pins=["L0.c1", "L2.h0", "L0.n1"])
    assert len(probe["spine"]) == 24 and len(probe["matrices"]["gate"]) == 11
    assert probe["final_lens_identity_error"] < 1e-6
    total = probe["dla"]["embedding_contribution"] + sum(r["operator_dla"] + r["mlp_dla"] for r in probe["spine"])
    assert abs(total - probe["margin"]) < 2e-6
    checks["spine_shapes_lens_identity_dla_additivity"] = True
    attr = run_gradient_attribution(lm, input_ids=ids, target=1, foil=2, full_scores=True)
    finite = {}
    for family, layer in (("h", 2), ("c", 0), ("n", 0)):
        unit = f"L{layer}.{family}1"
        def margin(scale):
            with torch.no_grad(), GlassboxHooks(model, 6, PatchSpec([
                    {"unit": unit, "op": "scale", "value": scale, "scope": "all"}])) as cap:
                model.model(ids)
                z = unembed(model, cap.points["out:23"][:, -1])[0]
                return float(z[1] - z[2])
        central = (margin(1.0001) - margin(0.9999)) / 0.0002
        analytic = attr["scores"][f"{family}:{layer}"][1]
        error = abs(central - analytic)
        assert error < 0.003 + 0.02 * abs(analytic), (unit, central, analytic)
        finite[unit] = {"analytic": analytic, "central_difference": central, "absolute_error": error}
    checks["one_pass_gradient_finite_differences"] = finite
    try:
        with GlassboxHooks(model, 6):
            raise RuntimeError("cleanup fixture")
    except RuntimeError:
        pass
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    assert all("forward" not in block.__dict__ for block in model.model.layers)
    assert all(p.grad is None for p in model.parameters())
    checks["exception_cleanup_no_parameter_gradients"] = True
    return {"passed": True, "checks": checks, "fixture": "24-layer float32 CPU hybrid, 8 hidden / 12 neurons / 2 heads"}
