"""Focused CPU/fake-backend integration checks. Never downloads model weights."""
from __future__ import annotations

import importlib
import importlib.util
import json
import time

from .schema import Budget, Case, Condition, Objective, Study, canonical_json


def schema_checks():
    study = Study("fixture", "Does a declared patch change a fixed token margin?",
                  "Identity controls preserve the measured margin.",
                  [Case("d0", "one", 1, 2), Case("h0", "two", 1, 2, split="holdout")],
                  [Condition("clean"), Condition("identity", {"rules": []})])
    study.validate()
    assert Study.from_dict(study.to_dict()).fingerprint == study.fingerprint
    def rejects(fn):
        try:
            fn()
        except (ValueError, TypeError):
            return
        raise AssertionError("Invalid protocol accepted")
    rejects(lambda: Case.from_dict({"id": "x", "prompt": "one", "unknown": 1}))
    rejects(lambda: Study.from_dict({**study.to_dict(), "prediction": ""}))
    rejects(lambda: Study.from_dict({**study.to_dict(), "baseline_id": "missing"}))
    rejects(lambda: Study.from_dict({**study.to_dict(), "cases": [study.cases[0].to_dict()] * 2}))
    rejects(lambda: Case("../unsafe", "one").validate())
    rejects(lambda: Case("x", "one", split="training").validate())
    rejects(lambda: Budget(max_trials=True).validate())
    rejects(lambda: Budget(max_seconds=float("nan")).validate())
    rejects(lambda: Objective("unknown").validate())
    rejects(lambda: Objective("token_margin", reduction="mean").validate())
    rejects(lambda: Study.from_dict({**study.to_dict(), "cases": [{"id":"x", "prompt":"one"}]}))
    rejects(lambda: Study.from_dict({**study.to_dict(), "conditions": [{"id":"clean", "input_variant":"corrupt"}]}))
    rejects(lambda: Condition("x", {"rules":[{"unit":"L2.h0", "op":"zero", "row":0}]}).validate())
    old = study.fingerprint
    study.conditions.append(Condition("new", {"rules": [{"unit":"L2.h0", "op":"zero"}]}))
    assert study.fingerprint != old
    canonical_json(study.to_dict())
    return {"passed": True, "roundtrip": True, "semantic_fingerprint": True,
            "invalid_protocols_rejected": 13, "local_model_loads": 0}


def run_checks():
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("CPU fixtures require PyTorch; install glass-box[instruments] (or .[instruments] from a checkout)")
    start = time.perf_counter()
    result = {"schema": schema_checks()}
    from lfm3lab.glassbox import synthetic_glassbox_checks
    result["glassbox"] = synthetic_glassbox_checks()
    for name in ("backend", "runner", "worker", "examples"):
        module = importlib.import_module(f"lfm3lab.research.checks_{name}")
        result[name] = module.run_checks()
        if result[name].get("passed") is False or result[name].get("ok") is False:
            raise AssertionError(f"{name} checks failed: {result[name]}")
    return {"passed": True, "mode": "CPU fixtures; no model weights", "checks": result,
            "elapsed_s": time.perf_counter() - start}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))
