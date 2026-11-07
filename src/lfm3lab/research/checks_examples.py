"""CPU-only example protocol checks. No weight loading or real inference."""
from __future__ import annotations

import copy
import importlib.util
from importlib.resources import files
import json
import math
from pathlib import Path
import re
import tempfile

from .schema import Study


def _reject(value):
    try:
        Study.from_dict(value)
    except (ValueError, TypeError):
        return
    raise AssertionError("Invalid example protocol was accepted")


def _patch_bounds(rule):
    """Check these examples' pinned coordinates without importing torch."""
    match = re.fullmatch(r"L(\d+)\.(?:([hcn])(\d+)|res\.(in|mid|out))", rule["unit"])
    assert match, rule
    layer, family, index, _ = match.groups()
    layer = int(layer)
    assert 0 <= layer < 24
    if family:
        assert 0 <= int(index) < {"h": 16, "c": 2048, "n": 7680}[family]
        conv = layer not in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
        assert family != "h" or not conv
        assert family != "c" or conv
    assert rule["op"] in ({"replace", "scale", "zero"} if not family else
                          {"scale", "zero"})
    assert rule.get("scope", "boundary") in ("all", "boundary", "answer", "prompt", "first4")
    if "value" in rule:
        assert type(rule["value"]) in (int, float) and math.isfinite(rule["value"])
    if rule["op"] == "replace":
        assert rule.get("source") == "clean"


def run_checks() -> dict:
    # Source checkouts keep examples at repo level; wheels bundle the same files
    # under the research package so `lfm3-research check` works after installation.
    source_examples = Path(__file__).resolve().parents[3] / "examples" / "mechanistic"
    bundled = files("lfm3lab.research").joinpath("examples")
    root = Path(str(bundled)) if bundled.is_dir() else source_examples
    paths = sorted(root.glob("*.json"))
    assert len(paths) == 2, "Expected exactly two distinct study examples"
    summaries = []
    kinds = set()
    for path in paths:
        raw = json.loads(path.read_text())
        study = Study.from_dict(raw)
        assert Study.from_dict(study.to_dict()).fingerprint == study.fingerprint
        kinds.add(study.objective.kind)
        counts = {split: sum(c.split == split for c in study.cases)
                  for split in ("discovery", "validation", "holdout")}
        assert all(n >= 2 for n in counts.values())
        assert len({c.prompt for c in study.cases}) == len(study.cases)
        assert study.question.strip() and study.prediction.strip()
        assert "no model findings" in study.metadata["status"]
        assert study.budget.max_trials == len(study.cases) * len(study.conditions)
        assert study.budget.max_input_tokens <= 256 and study.budget.max_seconds <= 600
        conditions = {c.id: c for c in study.conditions}
        baseline = conditions[study.baseline_id]
        assert baseline.input_variant == "clean" and not baseline.patch.get("rules", [])
        pairs = study.metadata["controls"]["identity_pairs"]
        assert pairs
        assert set(study.metadata["controls"]["no_op"]) == {p[0] for p in pairs}
        for identity_id, reference_id in pairs:
            identity, reference = conditions[identity_id], conditions[reference_id]
            assert identity.input_variant == reference.input_variant
            assert not reference.patch.get("rules", [])
            assert identity.patch.get("rules")
            assert all(r["op"] == "scale" and r["value"] == 1
                       for r in identity.patch["rules"])
        assert any(r["op"] != "scale" or r.get("value") != 1
                   for c in study.conditions for r in c.patch.get("rules", []))
        for condition in study.conditions:
            for rule in condition.patch.get("rules", []):
                _patch_bounds(rule)
                if rule.get("source") == "clean":
                    assert condition.input_variant == "corrupt"
                    assert all(c.corrupt_prompt and c.corrupt_prompt != c.prompt for c in study.cases)
        if study.objective.kind == "continuation_logprob":
            for case in study.cases:
                call = json.loads(case.continuation)
                assert isinstance(call["name"], str) and isinstance(call["arguments"], dict)
        invalid = copy.deepcopy(raw)
        invalid["unexpected"] = True
        _reject(invalid)
        invalid = copy.deepcopy(raw)
        invalid["cases"][0]["split"] = "test"
        _reject(invalid)
        invalid = copy.deepcopy(raw)
        invalid["conditions"][1]["patch"]["rules"][0]["value"] = float("nan")
        _reject(invalid)
        invalid = copy.deepcopy(raw)
        invalid["budget"]["max_input_tokens"] = 0
        _reject(invalid)
        invalid = copy.deepcopy(raw)
        required = "continuation" if study.objective.kind == "continuation_logprob" else "target"
        invalid["cases"][0].pop(required)
        _reject(invalid)
        summaries.append({"file": path.name, "fingerprint": study.fingerprint,
                          "splits": counts, "planned_trials": study.budget.max_trials})
    assert kinds == {"token_margin", "continuation_logprob"}

    runner_available = all(importlib.util.find_spec(f"lfm3lab.research.{name}")
                           for name in ("runner", "report", "session"))
    walkthrough = {"status": "skipped", "reason": "Runner/report/session not integrated yet"}
    if runner_available:
        spec = importlib.util.spec_from_file_location("mechanistic_fake_example", root / "fake_walkthrough.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="mechanistic-examples-check-") as directory:
            result = module.run_demo(directory, include_holdout=True)
        assert result["synthetic"]
        walkthrough = {"status": "passed", "synthetic": True,
                       "trial_calls": sum(s["trial_calls"] for s in result["studies"])}
    return {"ok": True, "model_free": True, "examples": summaries,
            "checks": ["schema_roundtrip", "fixed_objectives", "disjoint_prompts_and_splits",
                       "clean_and_identity_controls", "intervention_coordinates_and_budgets",
                       "malformed_protocol_rejection"], "walkthrough": walkthrough,
            "limits": ["Coordinate checks are pinned LFM3 bounds, not tokenizer validation.",
                       "Synthetic receipts demonstrate plumbing, not model effects."]}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))
