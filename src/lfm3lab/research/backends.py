"""Lazy, serialized adapter for the verified LFM3-1.7B Glass-Box primitives.

This module adds no eager torch/model-loader imports and status loads no weights.
The Glass-Box root initializer likewise loads no numerical dependencies.
The adapter owns every model it loads, and holds the process inference lock until
close has released the weights and caches. Public primitives remain unchanged.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import threading
import time
import traceback
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from .schema import Objective, canonical_json

PINNED = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
REPO_ID = "LiquidAI/LFM3-1.7B"
OPERATIONS = ("score", "probe", "attribute", "generate", "sweep")


def _implementation_identity():
    """Hash numerical helpers without importing torch, transformers or weights."""
    package = Path(__file__).resolve().parent.parent
    paths = ("research/backends.py", "research/session.py", "glassbox.py", "model.py", "jacobian_lens.py")
    sources = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in paths}
    runtime = {}
    for name in ("torch", "transformers"):
        try:
            runtime[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            runtime[name] = "not-installed"
    digest = hashlib.sha256(canonical_json({"source_sha256": sources, "runtime_versions": runtime}).encode()).hexdigest()
    return f"lfm3lab.research.LFM3Backend/torch-sdpa:sha256:{digest}", sources, runtime


def _device(value):
    value = str(value)
    if value == "cuda":
        return "cuda:0"
    if value in ("cpu", "mps") or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value):
        return value
    raise ValueError("device must be cpu, mps, cuda, or cuda:N")


def _dtype(value):
    value = str(value).removeprefix("torch.")
    if value not in ("bfloat16", "float32", "float16"):
        raise ValueError("dtype must be bfloat16, float32, or float16")
    return value


def _clear_exception_frames(exc):
    """Failed .to()/autograd frames can otherwise keep weights or graphs alive."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
            exc.__traceback__ = None
        next_exc = exc.__cause__ or exc.__context__
        exc.__cause__ = exc.__context__ = None
        exc = next_exc


class LFM3Backend:
    """One lazily loaded pinned instruct checkpoint, not an arbitrary LFM family.

    All calls, including status and close, serialize on a thread mutex. CPU/MPS
    weight loading is refused unless allow_local_model=True; CUDA is permitted.
    Explicit full commit revisions are required for stable unloaded/loaded IDs.
    """

    def __init__(self, device="cuda", revision=PINNED, dtype="bfloat16",
                 allow_local_model=False):
        self.device, self.dtype = _device(device), _dtype(dtype)
        if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            raise ValueError("revision must be an explicit 40-character checkpoint commit")
        if type(allow_local_model) is not bool:
            raise ValueError("allow_local_model must be boolean")
        self.revision = revision.lower()
        self.allow_local_model = allow_local_model
        self._implementation, self._source_hashes, self._runtime_versions = _implementation_identity()
        self._mutex = threading.RLock()
        self._loaded = None
        self._lifetime_lock = None
        self._closed = False

    def status(self):
        with self._mutex:
            return {"backend": "lfm3", "repo_id": REPO_ID, "revision": self.revision,
                    "adapter": "lfm3", "adapter_version": 1,
                    "implementation": self._implementation,
                    "implementation_version": 1,
                    "implementation_source_sha256": dict(self._source_hashes),
                    "runtime_versions": dict(self._runtime_versions),
                    "device": self.device, "dtype": self.dtype,
                    "loaded": self._loaded is not None, "closed": self._closed,
                    "capabilities": {
                        "operations": list(OPERATIONS), "objectives": [
                            "token_margin", "token_logprob", "continuation_logprob"],
                        "adapter": "LFM3-1.7B instruct pinned decoder API; not generic LFM support",
                        "verified_revision": PINNED,
                        "requested_revision_verified": self.revision == PINNED,
                        "coordinates": "L#.h#, L#.c#, L#.n#, L#.res.in/mid/out",
                        "norm_modes": ["native", "frozen"],
                        "frozen_norm_definition": "clean receiver post_operator_norm and post_ffn_norm RMS denominators only; other norms, including final norm, remain native",
                        "clean_residual_donor": ["score", "probe"],
                        "donor_alignment": "equal token counts and absolute predictor positions; no semantic alignment",
                        "attribute_objective": "target-minus-foil logit activation-times-gradient, not continuation attribution",
                        "generation": "greedy cached generation with full telemetry replay; clean-source donors unsupported",
                        "sweep_modes": ["heads", "layers", "mlp_layers", "blocks"],
                        "score_telemetry": False,
                    }}

    def _load_model(self):
        # Kept separate so model-free checks can supply tiny modules, not weights.
        import torch
        from lfm3lab.model import load
        return load("instruct", revision=self.revision, device=self.device,
                    dtype=getattr(torch, self.dtype))

    def _release_memory(self):
        from lfm3lab.glassbox import release_memory
        release_memory(self.device)

    def _ensure_loaded(self):
        if self._loaded is not None:
            return self._loaded
        if self.device in ("cpu", "mps") and not self.allow_local_model:
            raise PermissionError("CPU/MPS real model loading requires allow_local_model=True")
        from lfm3lab.glassbox import inference_lock
        lifetime = inference_lock()
        lifetime.__enter__()
        self._lifetime_lock = lifetime
        lm = None
        try:
            lm = self._load_model()
            parameter = next(lm.model.parameters())
            actual = (lm.repo_id, lm.revision.lower(), _device(parameter.device), _dtype(parameter.dtype))
            expected = (REPO_ID, self.revision, self.device, self.dtype)
            if actual != expected or _device(lm.device) != self.device:
                raise ValueError(f"Loaded model identity mismatch: {actual!r} != {expected!r}")
            self._loaded = lm
            return lm
        except BaseException as exc:
            # Remove locals in failed load frames before relinquishing ownership.
            _clear_exception_frames(exc)
            lm = None
            if "parameter" in locals():
                parameter = None
            self._loaded = None
            try:
                self._release_memory()
            except Exception:
                gc.collect()  # E.g. CUDA allocation failed before device init.
            finally:
                self._lifetime_lock = None
                lifetime.__exit__(None, None, None)
            raise exc from None

    def close(self):
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            had_model = self._loaded is not None
            self._loaded = None
            try:
                if had_model:
                    self._release_memory()
            finally:
                if self._lifetime_lock is not None:
                    lifetime, self._lifetime_lock = self._lifetime_lock, None
                    lifetime.__exit__(None, None, None)

    def _validate(self, operation, payload):
        if operation not in OPERATIONS:
            raise ValueError(f"Unsupported operation: {operation!r}")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        canonical_json(payload)
        common = {"prompt", "chat", "thinking", "target", "foil", "patch", "max_input_tokens"}
        extra = {
            "score": {"objective", "continuation", "donor_prompt"},
            "probe": {"position", "top_k", "pins", "routing", "input_ids", "prompt_length", "donor_prompt", "trace_start", "chosen_ids", "vocabulary_units"},
            "attribute": {"position", "top_k", "input_ids", "prompt_length", "full_scores"},
            "generate": {"max_new_tokens", "top_k", "pins"},
            "sweep": {"scope", "norm_mode", "batch_size", "units", "mode"},
        }
        if set(payload) - common - extra[operation]:
            raise ValueError(f"Unsupported {operation} fields: {sorted(set(payload) - common - extra[operation])}")
        if operation == "sweep" and "patch" in payload:
            raise ValueError("sweep accepts its own scope/norm_mode, not a patch")
        if "input_ids" not in payload and (not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip()):
            raise ValueError("A nonempty prompt is required")
        for name in ("chat", "thinking", "routing", "full_scores"):
            if name in payload and type(payload[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
        limit = payload.get("max_input_tokens", 256)
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ValueError("max_input_tokens must be 1..4096")
        if "input_ids" in payload:
            ids = payload["input_ids"]
            maximum = min(limit, 256 if operation == "attribute" else 768)
            if not isinstance(ids, list) or not 1 <= len(ids) <= maximum or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError("input_ids must be a bounded nonempty list of nonnegative token IDs")
        for name in ("position", "trace_start"):
            if payload.get(name) is not None and (type(payload[name]) is not int or payload[name] < 0):
                raise ValueError(f"{name} must be a nonnegative absolute predictor position")
        if "prompt_length" in payload and (type(payload["prompt_length"]) is not int or payload["prompt_length"] < 1):
            raise ValueError("prompt_length must be a positive integer")
        if payload.get("chosen_ids") is not None and (not isinstance(payload["chosen_ids"], list) or any(type(i) is not int or i < 0 for i in payload["chosen_ids"])):
            raise ValueError("chosen_ids must be nonnegative token IDs")
        from lfm3lab.glassbox import PatchSpec, Unit, SCOPES
        spec = PatchSpec.from_value(payload.get("patch"))
        if any("row" in r for r in spec.rules):
            raise ValueError("Batch row coordinates are reserved for native sweeps")
        donor = any(r.get("source") == "clean" for r in spec.rules)
        if donor and operation not in ("score", "probe"):
            raise ValueError(f"{operation} does not support clean-source residual donors")
        if donor and (not isinstance(payload.get("donor_prompt"), str) or not payload["donor_prompt"].strip()):
            raise ValueError("clean-source residual replacement requires explicit donor_prompt")
        if "donor_prompt" in payload and not donor:
            raise ValueError("donor_prompt requires a clean-source residual replacement")
        for name in ("target", "foil"):
            value = payload.get(name)
            if value is not None and not (type(value) is int and value >= 0 or isinstance(value, str) and value):
                raise ValueError(f"{name} must be a nonnegative ID or exact one-token text")
        if operation == "score":
            objective = Objective.from_dict(payload.get("objective", {}))
            if objective.kind.startswith("token_") and payload.get("target") is None:
                raise ValueError("Token scoring requires a fixed target")
            if objective.kind == "token_margin" and (payload.get("foil") is None or payload["target"] == payload["foil"]):
                raise ValueError("Token margin requires distinct fixed target and foil")
            if objective.kind == "continuation_logprob" and (not isinstance(payload.get("continuation"), str) or not payload["continuation"]):
                raise ValueError("Continuation scoring requires explicit nonempty continuation text")
        if "top_k" in payload and (type(payload["top_k"]) is not int or not 1 <= payload["top_k"] <= 64):
            raise ValueError("top_k must be 1..64")
        for name, maximum in (("pins", 32), ("vocabulary_units", 4)):
            if name in payload:
                if not isinstance(payload[name], list) or len(payload[name]) > maximum:
                    raise ValueError(f"{name} must be a list of at most {maximum} units")
                for address in payload[name]:
                    unit = Unit.parse(address)
                    if name == "vocabulary_units" and unit.family not in ("c", "n"):
                        raise ValueError("Vocabulary projections require c/n units")
        if operation == "generate":
            count = payload.get("max_new_tokens", 32)
            if type(count) is not int or not 1 <= count <= 128:
                raise ValueError("max_new_tokens must be 1..128")
        if operation == "sweep":
            size = payload.get("batch_size", 4)
            mode = payload.get("mode", "heads")
            if type(size) is not int or not 1 <= size <= 16 or mode not in ("heads", "layers", "mlp_layers", "blocks"):
                raise ValueError("Invalid sweep mode or batch_size (1..16)")
            if payload.get("scope", "boundary") not in SCOPES or payload.get("norm_mode", "native") not in ("native", "frozen"):
                raise ValueError("Invalid sweep scope or norm_mode")
            if payload.get("units") is not None:
                units = payload["units"]
                if mode != "heads" or not isinstance(units, list) or not units or len(set(units)) != len(units):
                    raise ValueError("units requires unique head addresses in heads mode")
                if any(Unit.parse(u).family != "h" for u in units):
                    raise ValueError("Head sweep requires h units")
        return spec

    def run(self, operation, payload):
        with self._mutex:
            if self._closed:
                raise RuntimeError("Backend is closed")
            if isinstance(payload, dict):
                payload = json.loads(canonical_json(payload))  # Detached request, including nested patch rules.
            self._validate(operation, payload)
            lm = self._ensure_loaded()
            try:
                result = self._score(lm, payload) if operation == "score" else self._primitive(lm, operation, payload)
                result.update(repo_id=REPO_ID, revision=self.revision, device=self.device, dtype=self.dtype)
                result.update(implementation=self._implementation, runtime_versions=dict(self._runtime_versions))
                canonical_json(result)
                return result
            except BaseException as exc:
                _clear_exception_frames(exc)
                # Autograd and cached decode failures must not retain old graphs.
                try:
                    self._release_memory()
                except Exception:
                    gc.collect()
                raise exc from None

    def _score_reference(self, lm, ids, prompt_length, spec, donor_ids):
        """Capture only sandwich denominators and explicitly requested points."""
        import torch
        from lfm3lab.glassbox import GlassboxHooks, Unit
        reference = {"inv": {}, "points": {}}
        if spec.norm_mode == "frozen":
            with torch.no_grad(), GlassboxHooks(lm.model, prompt_length, capture=False) as clean:
                lm.model.model(input_ids=ids, use_cache=False)
                reference["inv"] = {k: v.detach().clone() for k, v in clean.inv.items()}
        keys = {Unit.parse(r["unit"], lm.model.config).checkpoint for r in spec.rules if r.get("source") == "clean"}
        if keys:
            with torch.no_grad(), GlassboxHooks(lm.model, prompt_length, capture=False) as donor:
                for key in keys:
                    boundary, layer = key.split(":")
                    block = lm.model.model.layers[int(layer)]
                    def save_pre(m, args, k=key):
                        reference["points"][k] = args[0].detach().clone()
                    def save_out(m, args, output, k=key):
                        reference["points"][k] = output.detach().clone()
                    donor.add(block.ffn_norm if boundary == "mid" else block,
                              save_out if boundary == "out" else save_pre, pre=boundary != "out")
                lm.model.model(input_ids=donor_ids, use_cache=False)
        return reference

    def _score(self, lm, payload):
        import torch
        from lfm3lab.glassbox import GlassboxHooks, PatchSpec, prepare_ids
        from lfm3lab.jacobian_lens import sync
        objective = Objective.from_dict(payload.get("objective", {}))
        limit = payload.get("max_input_tokens", 256)
        chat, thinking = payload.get("chat", False), payload.get("thinking", False)
        ids = prepare_ids(lm, payload["prompt"], chat=chat, thinking=thinking, max_tokens=limit)
        prompt_length = ids.shape[1]
        spec = PatchSpec.from_value(payload.get("patch"), lm.model.config)
        donor_ids = None
        if any(r.get("source") == "clean" for r in spec.rules):
            donor_ids = prepare_ids(lm, payload["donor_prompt"], chat=chat, thinking=thinking, max_tokens=limit)
            if donor_ids.shape != ids.shape:
                raise ValueError("Clean donor and receiver prompts must have identical token counts")
        vocab = lm.model.lm_head.weight.shape[0]
        def token_id(value):
            values = [value] if type(value) is int else lm.tokenizer.encode(value, add_special_tokens=False)
            if len(values) != 1 or type(values[0]) is not int or not 0 <= values[0] < vocab:
                raise ValueError(f"Expected exact one-token text or vocabulary ID: {value!r} -> {values!r}")
            return values[0]
        target_id = token_id(payload["target"]) if objective.kind.startswith("token_") else None
        foil_id = token_id(payload["foil"]) if objective.kind == "token_margin" else None
        if objective.kind == "token_margin" and target_id == foil_id:
            raise ValueError("Resolved target and foil must differ")
        continuation_ids = []
        if objective.kind == "continuation_logprob":
            continuation_ids = lm.tokenizer.encode(payload["continuation"], add_special_tokens=False)
            if not continuation_ids or any(type(i) is not int or not 0 <= i < vocab for i in continuation_ids):
                raise ValueError("Continuation must encode to nonempty valid token IDs")
            tail = torch.tensor([continuation_ids[:-1]], dtype=ids.dtype, device=ids.device)
            ids = torch.cat([ids, tail], dim=1)
            if donor_ids is not None:
                donor_ids = torch.cat([donor_ids, tail], dim=1)
            if ids.shape[1] > limit:
                raise ValueError(f"Teacher-forced predictor sequence has {ids.shape[1]} tokens; budget is {limit}")
        sync(lm.device)
        start = time.perf_counter()
        with torch.no_grad():
            reference = self._score_reference(lm, ids, prompt_length, spec, donor_ids)
            # Native/no-rule scoring is a plain body pass, with no telemetry hooks.
            if spec.rules or spec.norm_mode == "frozen":
                with GlassboxHooks(lm.model, prompt_length, spec, reference=reference, capture=False):
                    hidden = lm.model.model(input_ids=ids, use_cache=False).last_hidden_state
            else:
                hidden = lm.model.model(input_ids=ids, use_cache=False).last_hidden_state
            positions = list(range(prompt_length - 1, ids.shape[1]))
            logprobs = []
            if objective.kind == "continuation_logprob":
                for begin in range(0, len(positions), 16):
                    chunk = positions[begin:begin + 16]
                    logits = lm.model.lm_head(hidden[0, chunk]).float()
                    tokens = torch.tensor(continuation_ids[begin:begin + len(chunk)], device=logits.device)
                    values = logits.log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
                    logprobs.extend(values.cpu().tolist())
                value = sum(logprobs) / (len(logprobs) if objective.reduction == "mean" else 1)
                definition = f"{objective.reduction} of fp32 log_softmax(native lm_head logits)[continuation_id] at teacher-forced predictor positions, with Python binary64 accumulation"
            else:
                logits = lm.model.lm_head(hidden[:, -1])[0].float()
                if objective.kind == "token_margin":
                    value = float(logits[target_id] - logits[foil_id])
                    definition = "fp32(native lm_head logit[target_id]) minus fp32(native lm_head logit[foil_id]) at last prompt token"
                else:
                    value = float(logits.log_softmax(-1)[target_id])
                    definition = "fp32 log_softmax(native lm_head logits)[target_id] at last prompt token"
        if not math.isfinite(value):
            raise ValueError("Scoring produced a nonfinite value")
        sync(lm.device)
        result = {"kind": "score", "value": value, "objective": objective.to_dict(),
                  "revision": lm.revision, "device": self.device, "dtype": self.dtype,
                  "input_ids": ids[0].tolist(), "prompt_length": prompt_length,
                  "elapsed_s": time.perf_counter() - start, "definition": definition,
                  "predictor_positions": positions, "target_id": target_id, "foil_id": foil_id,
                  "patch": asdict(spec), "reference_passes": int(spec.norm_mode == "frozen") + int(donor_ids is not None)}
        if continuation_ids:
            result.update(continuation_ids=continuation_ids, token_logprobs=logprobs,
                          continuation_tokenization="encode(continuation, add_special_tokens=False), appended to separately encoded prompt; no text-boundary retokenization")
        if donor_ids is not None:
            result["donor_input_ids"] = donor_ids[0].tolist()
        return result

    def _primitive(self, lm, operation, payload):
        from lfm3lab import glassbox as gb
        kwargs = dict(payload)
        limit = kwargs.pop("max_input_tokens", 256)
        kwargs.setdefault("chat", False)
        native_limit = 256 if operation in ("attribute", "sweep") else 512
        if operation == "probe" and "input_ids" in kwargs:
            native_limit = 768
        ids = gb.explicit_ids(lm, kwargs["input_ids"], min(limit, native_limit)) if "input_ids" in kwargs else gb.prepare_ids(
            lm, kwargs.get("prompt"), chat=kwargs["chat"], thinking=kwargs.get("thinking", False), max_tokens=min(limit, native_limit))
        if kwargs.get("donor_prompt") is not None:
            donor_ids = gb.prepare_ids(lm, kwargs["donor_prompt"], chat=kwargs["chat"],
                                       thinking=kwargs.get("thinking", False), max_tokens=min(limit, 512))
            if donor_ids.shape != ids.shape:
                raise ValueError("Clean donor and receiver must have identical token counts")
        if operation == "probe" and kwargs.get("chosen_ids") is not None:
            trace_start = kwargs.get("trace_start", 0)
            if not 0 <= trace_start < ids.shape[1] or len(kwargs["chosen_ids"]) != ids.shape[1] - trace_start:
                raise ValueError("chosen_ids must match the requested predictor trace length")
        if operation in ("probe", "attribute"):
            kwargs["input_ids"] = ids
        functions = {"probe": gb.run_glassbox_probe, "attribute": gb.run_gradient_attribution,
                     "generate": gb.run_glassbox_generate, "sweep": gb.run_head_ablation_sweep}
        return functions[operation](lm, **kwargs)
