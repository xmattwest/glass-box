"""CPU fixtures for adapter math, ownership, serialization and session protocol.

No Hugging Face calls, real checkpoint weights, or CUDA execution. Run with
``uv run python -m lfm3lab.research.checks_backend``.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import threading
from tempfile import TemporaryDirectory
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from .backends import LFM3Backend, PINNED, REPO_ID
from .schema import Objective
from .session import ResearchSession


def _tiny(dtype="float32"):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.linspace(0.7, 1.3, 8))
            self.variance_epsilon = 1e-6

        def forward(self, x):
            inv = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.variance_epsilon)
            return self.weight * (x.float() * inv).to(x.dtype)

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
            self.conv = nn.Conv1d(8, 8, 3, padding=2, groups=8, bias=False)

        def forward(self, hidden_states=None, **kwargs):
            b, c, x = self.in_proj(hidden_states).chunk(3, -1)
            value = self.conv((b * x).transpose(1, 2))[..., :hidden_states.shape[1]].transpose(1, 2)
            return self.out_proj(c * value)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.v_proj, self.o_proj = nn.Linear(8, 2, False), nn.Linear(8, 8, False), nn.Linear(8, 8, False)

        def forward(self, hidden_states=None, **kwargs):
            x = hidden_states
            v = self.v_proj(x).cumsum(1)
            v = v / torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)[None, :, None]
            gates = self.gate_proj(x).sigmoid()[..., None]
            return self.o_proj((v.view(*x.shape[:2], 2, 4) * gates).flatten(-2)), None

    class Block(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.operator_norm, self.post_operator_norm = Norm(), Norm()
            self.ffn_norm, self.post_ffn_norm = Norm(), Norm()
            self.feed_forward = MLP()
            if kind == "conv":
                self.conv = Conv()
            else:
                self.self_attn = Attention()

        def forward(self, hidden_states, **kwargs):
            x = self.operator_norm(hidden_states)
            value = self.conv(hidden_states=x) if self.kind == "conv" else self.self_attn(hidden_states=x)[0]
            hidden_states = hidden_states + self.post_operator_norm(value)
            return hidden_states + self.post_ffn_norm(self.feed_forward(self.ffn_norm(hidden_states)))

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block("conv"), Block("conv"), Block("full_attention")])
            self.embed_tokens, self.embedding_norm = nn.Embedding(32, 8), Norm()
            self.calls = 0

        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            self.calls += 1
            x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            for block in self.layers:
                x = block(x)
            return SimpleNamespace(last_hidden_state=self.embedding_norm(x))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(layer_types=["conv", "conv", "full_attention"],
                                          hidden_size=8, intermediate_size=12, head_dim=4,
                                          num_attention_heads=2, _attn_implementation="sdpa")
            self.model, self.lm_head = Body(), nn.Linear(8, 32, False)

    class Tokenizer:
        eos_token_id = 31

        def encode(self, text, **kwargs):
            # Deliberate text-boundary merge: continuation must NOT use this.
            if text == "abX":
                return [20]
            return [ord(c) % 31 for c in text]

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([self.encode(text)], dtype=torch.long)}

        def decode(self, ids, **kwargs):
            return ":".join(str(i) for i in ids)

        def apply_chat_template(self, messages, **kwargs):
            return "Q" + messages[0]["content"] + "A"

    with torch.random.fork_rng():
        torch.manual_seed(2401)
        model = Tiny().eval().to(dtype=getattr(torch, dtype))
    return SimpleNamespace(model=model, tokenizer=Tokenizer(), repo_id=REPO_ID,
                           revision=PINNED, device="cpu")


class _FixtureBackend(LFM3Backend):
    def __init__(self, dtype="float32", events=None):
        super().__init__(device="cpu", dtype=dtype, allow_local_model=True)
        self.events = events if events is not None else []
        self.loads = 0

    def _load_model(self):
        self.events.append("load")
        self.loads += 1
        return _tiny(self.dtype)

    def _release_memory(self):
        gc.collect()
        self.events.append("release")


def _raises(error, function):
    try:
        function()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def run_identity_checks():
    """Focused metadata/provenance check; no model or fixture inference."""
    def make(torch_version, transformers_version):
        versions = {"torch": torch_version, "transformers": transformers_version}
        with patch("lfm3lab.research.backends.metadata.version", side_effect=versions.__getitem__) as lookup:
            backend = LFM3Backend()
            assert lookup.call_count == 2
            unloaded = backend.status()
            backend._loaded = object()  # State-only sentinel, not a model.
            loaded = backend.status()
            backend._loaded = None
            assert loaded["loaded"] and not unloaded["loaded"]
            for key in ("implementation", "implementation_source_sha256", "runtime_versions"):
                assert loaded[key] == unloaded[key]
            assert lookup.call_count == 2  # Runtime identity is frozen at construction.
            with patch.object(backend, "_ensure_loaded", return_value=object()), patch.object(
                    backend, "_score", return_value={"value": 0}):
                result = backend.run("score", {"prompt": "fixture", "target": 1, "foil": 2})
                assert result["implementation"] == unloaded["implementation"]
                assert result["runtime_versions"] == versions
            backend.close()
            return unloaded
    first = make("2.14.0", "5.21.0")
    repeat = make("2.14.0", "5.21.0")
    torch_changed = make("2.9.1+cu128", "5.21.0")
    transformers_changed = make("2.14.0", "5.22.0")
    assert first["implementation"] == repeat["implementation"]
    assert first["implementation"] != torch_changed["implementation"]
    assert first["implementation"] != transformers_changed["implementation"]
    for status in (repeat, torch_changed, transformers_changed):
        assert status["implementation_source_sha256"] == first["implementation_source_sha256"]
    assert all(len(value) == 64 for value in first["implementation_source_sha256"].values())
    return {"passed": True, "runtime_version_changes_identity": True,
            "source_hashes_unchanged_by_runtime": True, "unloaded_loaded_identity_stable": True,
            "score_runtime_provenance": True, "model_loads": 0, "inference_calls": 0}


def run_checks():
    import torch
    from lfm3lab.glassbox import GlassboxHooks, run_glassbox_probe

    checks = {}
    checks["runtime_identity_and_score_provenance"] = run_identity_checks()
    # Invalid or disallowed requests must never reach a loading function.
    backend = LFM3Backend(device="cpu")
    with patch.object(backend, "_load_model", side_effect=AssertionError("real weights forbidden")) as loader:
        assert backend.status()["loaded"] is False
        _raises(PermissionError, lambda: backend.run("score", {"prompt": "ab", "target": 1, "foil": 2}))
        invalid = [
            ("score", {"prompt": "ab", "target": 1}),
            ("score", {"prompt": "ab", "target": 1, "foil": 1}),
            ("score", {"prompt": "ab", "target": True, "foil": 2}),
            ("score", {"prompt": "ab", "target": 1, "foil": 2, "unknown": 1}),
            ("score", {"prompt": "ab", "objective": {"kind": "continuation_logprob"}}),
            ("score", {"prompt": "ab", "target": 1, "foil": 2, "patch": {"rules": [{"unit": "L0.h0", "op": "zero"}]}}),
            ("attribute", {"prompt": "ab", "patch": {"rules": [{"unit": "L0.res.mid", "op": "replace", "source": "clean"}]}}),
            ("generate", {"prompt": "ab", "max_new_tokens": 129}),
            ("generate", {"prompt": "ab", "patch": {"rules": [{"unit": "L0.res.mid", "op": "replace", "source": "clean"}]}}),
            ("sweep", {"prompt": "ab", "mode": "arbitrary"}),
            ("probe", {"input_ids": [1, -1]}),
            ("score", {"prompt": "ab", "target": 1, "foil": 2, "patch": {"rules": [{"unit": "L0.c1", "op": "scale", "value": float("nan")}]}}),
        ]
        for operation, payload in invalid:
            _raises(ValueError, lambda o=operation, p=payload: backend.run(o, p))
        assert loader.call_count == 0
    backend.close()
    assert LFM3Backend().status()["device"] == "cuda:0"
    assert LFM3Backend(dtype="torch.bfloat16").status()["dtype"] == "bfloat16"
    _raises(ValueError, lambda: LFM3Backend(revision="main"))
    checks["lazy_status_and_preload_rejection"] = len(invalid)

    events = []
    @contextmanager
    def recording_lock():
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    with patch("lfm3lab.glassbox.inference_lock", recording_lock):
        backend = _FixtureBackend(events=events)
        before = backend.status()
        session = ResearchSession(backend)
        base = {"prompt": "abcd", "target": 1, "foil": 2}
        margin = session.score(**base)
        lm = backend._loaded
        assert lm is not None
        after = backend.status()
        for key in ("backend", "repo_id", "revision", "device", "dtype", "capabilities",
                    "adapter", "adapter_version", "implementation", "implementation_version"):
            assert before[key] == after[key]
        from .store import backend_identity
        assert backend_identity(before) == backend_identity(after)
        ids = torch.tensor([lm.tokenizer.encode(base["prompt"])])
        with torch.no_grad():
            z = lm.model.lm_head(lm.model.model(input_ids=ids).last_hidden_state[:, -1])[0].float()
        assert abs(margin["value"] - float(z[1] - z[2])) < 1e-7
        text_margin = session.score(prompt="abcd", target="a", foil="b")
        assert text_margin["target_id"] == lm.tokenizer.encode("a")[0]
        _raises(ValueError, lambda: session.score(prompt="abcd", target="ab", foil="c"))
        _raises(ValueError, lambda: session.score(prompt="abcd", target=32, foil=2))
        _raises(ValueError, lambda: session.score(prompt="abcd", target="a", foil=lm.tokenizer.encode("a")[0]))
        lp = session.score(prompt="abcd", target=1, objective=Objective(kind="token_logprob"))
        assert abs(lp["value"] - float(z.log_softmax(-1)[1])) < 1e-7
        # Score must not call the full telemetry primitive, nor capture activations.
        with patch("lfm3lab.glassbox.run_glassbox_probe", side_effect=AssertionError("telemetry in score")), patch.object(
                GlassboxHooks, "reference_copy", side_effect=AssertionError("full reference in score")):
            called_before = lm.model.model.calls
            session.score(**base)
            assert lm.model.model.calls - called_before == 1
            patched = session.score(**base, patch={"rules": [{"unit": "L0.c1", "op": "zero"}]})
            assert patched["value"] != margin["value"]
            frozen = session.score(**base, patch={"norm_mode": "frozen"})
            assert frozen["value"] == margin["value"]
            for address in ("L0.res.in", "L0.res.mid", "L0.res.out", "L1.res.in", "L2.res.mid"):
                restored = session.score(prompt="zbcd", target=1, foil=2, donor_prompt="abcd",
                                         patch={"rules": [{"unit": address, "op": "replace", "source": "clean", "scope": "all"}]})
                assert abs(restored["value"] - margin["value"]) < 1e-7, address
                assert restored["donor_input_ids"] == margin["input_ids"]
                assert restored["reference_passes"] == 1
        with torch.no_grad():
            direct = run_glassbox_probe(lm, "abcd", target=1, foil=2, chat=False, top_k=1)
        assert abs(direct["margin"] - margin["value"]) < 1e-7
        _raises(ValueError, lambda: session.score(prompt="abcd", target=1, foil=2, donor_prompt="a",
                                                patch={"rules": [{"unit": "L0.res.mid", "op": "replace", "source": "clean"}]}))
        checks["fixed_id_margin_logprob_and_probe_agreement"] = True
        checks["score_without_telemetry_actual_residual_transplants"] = True

        # Explicit continuation IDs use predictors n-1 .. n+m-2, not targets n..
        continuation = "XYZ" * 7  # More than one vocabulary chunk.
        kwargs = {"prompt": "ab", "continuation": continuation, "objective": {"kind": "continuation_logprob"}}
        total = session.score(**kwargs)
        mean = session.score(**{**kwargs, "objective": {"kind": "continuation_logprob", "reduction": "mean"}})
        assert total["prompt_length"] == 2
        assert total["input_ids"] == lm.tokenizer.encode("ab") + lm.tokenizer.encode(continuation)[:-1]
        assert total["predictor_positions"] == list(range(1, 22))
        with torch.no_grad():
            h = lm.model.model(input_ids=torch.tensor([total["input_ids"]])).last_hidden_state
            expected = lm.model.lm_head(h[0, total["predictor_positions"]]).float().log_softmax(-1)
            expected = expected.gather(-1, torch.tensor(total["continuation_ids"])[:, None]).squeeze(-1)
        torch.testing.assert_close(torch.tensor(total["token_logprobs"]), expected)
        assert abs(total["value"] - float(expected.double().sum())) < 1e-6
        assert abs(mean["value"] * 21 - total["value"]) < 1e-12
        single = session.score(prompt="ab", continuation="X", objective={"kind": "continuation_logprob"})
        assert single["input_ids"] == lm.tokenizer.encode("ab")
        assert single["continuation_ids"] == lm.tokenizer.encode("X")
        assert single["input_ids"] != lm.tokenizer.encode("abX")
        _raises(ValueError, lambda: session.score(**kwargs, max_input_tokens=21))
        assert session.score(**kwargs, max_input_tokens=22)["input_ids"] == total["input_ids"]
        checks["teacher_forcing_predictors_boundary_tokenization_chunked_sum_mean"] = 21

        # Frozen receiver denominators must not come from the distinct donor.
        donor_patch = {"norm_mode": "frozen", "rules": [{"unit": "L0.res.mid", "op": "replace", "source": "clean", "scope": "all"}]}
        score = session.score(prompt="zbcd", donor_prompt="abcd", target=1, foil=2, patch=donor_patch)
        probe = session.probe(prompt="zbcd", donor_prompt="abcd", target=1, foil=2, patch=donor_patch, top_k=1)
        assert abs(score["value"] - probe["margin"]) < 1e-7
        assert score["reference_passes"] == 2
        checks["frozen_receiver_norms_distinct_from_donor"] = True
        # Dispatch smoke for differentiable attribution and native ablation.
        attr = session.attribute(**base, top_k=1)
        assert attr["kind"] == "attribute" and attr["backward_passes"] == 1
        sweep = session.sweep(**base, units=["L2.h0"], batch_size=1)
        assert len(sweep["rows"]) == 1
        with patch("lfm3lab.glassbox.run_glassbox_generate", return_value={"kind": "generate"}) as generation:
            assert session.generate(prompt="abcd", max_new_tokens=2)["kind"] == "generate"
            assert generation.call_args.kwargs == {"prompt": "abcd", "max_new_tokens": 2, "chat": False}
        checks["attribute_and_sweep_dispatch"] = True
        checks["generation_dispatch_only_not_cached_inference"] = True
        assert backend.loads == 1
        session.close()
        session.close()
        assert events[0:2] == ["lock", "load"] and events[-2:] == ["release", "unlock"]
        assert events.count("unlock") == 1
        _raises(RuntimeError, lambda: session.score(**base))
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in lm.model.modules())
        assert all("forward" not in b.__dict__ for b in lm.model.model.layers)
        assert all(p.grad is None for p in lm.model.parameters())
        checks["lifetime_lock_load_once_cleanup_and_identity_stability"] = True

    # BF16 catches the norm multiply/cast ordering that float32 cannot expose.
    with patch("lfm3lab.glassbox.inference_lock", recording_lock):
        backend = _FixtureBackend(dtype="bfloat16")
        with ResearchSession(backend) as session:
            native = session.score(**base)
            frozen = session.score(**base, patch={"norm_mode": "frozen"})
            assert native["value"] == frozen["value"]
            assert native["dtype"] == "bfloat16"
            lm = backend._loaded
            ids = torch.tensor([lm.tokenizer.encode(base["prompt"])])
            with torch.no_grad(), GlassboxHooks(lm.model, 4) as clean:
                lm.model.model(input_ids=ids)
                references = clean.reference_copy()
            with torch.no_grad(), GlassboxHooks(lm.model, 4, {"norm_mode": "frozen"}, reference=references) as cap:
                lm.model.model(input_ids=ids)
                for key, point in cap.points.items():
                    torch.testing.assert_close(point, references["points"][key], atol=0, rtol=0)
            checks["bfloat16_frozen_identity_all_residuals"] = True

    # Exception frames from partially loaded weights are dropped BEFORE unlock.
    failures = []
    class FailedBackend(_FixtureBackend):
        def _load_model(self):
            partial = _tiny()
            self.ref = weakref.ref(partial.model)
            failures.append("allocated")
            raise RuntimeError("synthetic load failure")

        def _release_memory(self):
            gc.collect()
            assert self.ref() is None
            failures.append("released")

    @contextmanager
    def failed_lock():
        failures.append("lock")
        try:
            yield
        finally:
            assert failures[-1] == "released"
            failures.append("unlock")

    with patch("lfm3lab.glassbox.inference_lock", failed_lock):
        backend = FailedBackend()
        _raises(RuntimeError, lambda: backend.run("score", base))
        assert not backend.status()["loaded"] and backend._lifetime_lock is None
        backend.close()
    assert failures == ["lock", "allocated", "released", "unlock"]
    checks["failed_load_releases_exception_graph_before_unlock"] = True

    class GraphFailureBackend(_FixtureBackend):
        def _score(self, lm, payload):
            graph = torch.randn(4, requires_grad=True).square()
            self.ref = weakref.ref(graph)
            raise RuntimeError("synthetic graph failure")

        def _release_memory(self):
            gc.collect()
            if hasattr(self, "ref"):
                assert self.ref() is None

    with patch("lfm3lab.glassbox.inference_lock", recording_lock):
        backend = GraphFailureBackend()
        _raises(RuntimeError, lambda: backend.run("score", base))
        assert backend.status()["loaded"]
        backend.close()
    checks["failed_operation_drops_graph_keeps_owned_model"] = True

    # A blocking fixture detects simultaneous scoring, double load, and close
    # while another thread is still using the model, without time-based sleeps.
    entered, leave = threading.Event(), threading.Event()
    serial = []
    class SerialBackend(_FixtureBackend):
        def _score(self, lm, payload):
            serial.append("start")
            if len(serial) == 1:
                entered.set()
                assert leave.wait(10), "serialization fixture timed out"
            serial.append("end")
            return {"value": 1, "objective": Objective().to_dict(), "input_ids": [1], "prompt_length": 1, "elapsed_s": 0}

    with patch("lfm3lab.glassbox.inference_lock", recording_lock):
        backend = SerialBackend()
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(backend.run, "score", base)
            assert entered.wait(10)
            second = pool.submit(backend.run, "score", base)
            assert backend.loads == 1
            leave.set()
            first.result(timeout=10)
            second.result(timeout=10)
        assert serial == ["start", "end", "start", "end"] and backend.loads == 1
        backend.close()
    checks["thread_serialization_single_load"] = True
    entered.clear()
    leave.clear()
    serial.clear()
    closing = threading.Event()
    with patch("lfm3lab.glassbox.inference_lock", recording_lock):
        backend = SerialBackend(events=events)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(backend.run, "score", base)
            assert entered.wait(10)
            def close_from_thread():
                closing.set()
                backend.close()
            closer = pool.submit(close_from_thread)
            assert closing.wait(10)
            assert backend._loaded is not None and not backend._closed
            leave.set()
            first.result(timeout=10)
            closer.result(timeout=10)
        assert serial == ["start", "end"] and backend.status()["closed"]
        assert backend.events[-2:] == ["release", "unlock"]
    checks["close_waits_for_active_thread_before_unlock"] = True

    # Exercise the actual evidence runner with actual adapter scoring, rather
    # than a canned response fake. A newly unloaded adapter resumes without load.
    from .runner import run_study
    from .schema import Budget, Case, Condition, Study
    with TemporaryDirectory(prefix="lfm3-backend-check-") as temporary, patch(
            "lfm3lab.glassbox.inference_lock", recording_lock):
        for kind in ("token_margin", "token_logprob", "continuation_logprob"):
            case = Case("tiny", "abcd", target=1 if kind != "continuation_logprob" else None,
                        foil=2 if kind == "token_margin" else None,
                        continuation="XYZ" if kind == "continuation_logprob" else None)
            study = Study(f"backend-{kind}", "Do adapter receipts survive the actual evidence pipeline?",
                          "Both paired fixture receipts are valid and an unloaded adapter resumes without inference.",
                          [case], [Condition("clean"), Condition("identity", {"norm_mode": "frozen"})],
                          objective=Objective(kind), budget=Budget(max_trials=4, max_seconds=30, max_input_tokens=16))
            output = Path(temporary) / kind
            backend = _FixtureBackend()
            with ResearchSession(backend) as session:
                report = run_study(study, session, output)
                assert report["splits"]["discovery"]["status_counts"] == {"success": 2, "failed": 0, "interrupted": 0, "missing": 0}
                assert report["splits"]["discovery"]["comparisons"]["identity"]["delta"]["mean"] == 0
                assert backend.loads == 1
            backend = _FixtureBackend()
            with ResearchSession(backend) as session:
                resumed = run_study(study, session, output)
                assert resumed["budget_used"]["attempts"] == 2
                assert backend.loads == 0
    checks["real_runner_all_objectives_unloaded_resume_no_failed_rows"] = 3

    class Echo:
        def __init__(self):
            self.closes = 0
        def run(self, operation, payload):
            return {"operation": operation, "payload": payload}
        def status(self):
            return {"fixture": True}
        def close(self):
            self.closes += 1

    echo = Echo()
    with ResearchSession(echo) as session:
        for operation in ("score", "probe", "attribute", "generate", "sweep"):
            assert getattr(session, operation)(prompt="fixture")["operation"] == operation
        assert session.status() == {"fixture": True}
    session.close()
    assert echo.closes == 1
    with patch("lfm3lab.research.worker.RemoteBackend", return_value=Echo()) as remote:
        with ResearchSession.remote("http://127.0.0.1:12345", token="fixture") as session:
            assert session.probe(prompt="fixture")["operation"] == "probe"
        remote.assert_called_once_with("http://127.0.0.1:12345", token="fixture")
    _raises(TypeError, lambda: ResearchSession(object()))
    checks["session_context_convenience_and_idempotent_close"] = True
    return {"passed": True, "checks": checks,
            "fixture": "3-layer 8-hidden CPU hybrid, float32 and bfloat16; no real model weights"}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))
