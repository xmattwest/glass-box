"""Exercise saved study receipts with invented CPU scores, never real weights.

This fake tests orchestration, not model behavior. It is intentionally not a
tokenizer or implementation of attention, convolution, or residual dynamics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from lfm3lab.research.schema import Study

EXAMPLE_PATHS = (Path(__file__).with_name("code_retrieval.json"),
                 Path(__file__).with_name("tool_continuation.json"))


class FakeBackend:
    """Minimal documented backend contract; synthetic identities are stable."""

    def __init__(self):
        self.closed = False
        self.calls = []

    def status(self):
        return {"backend": "synthetic-example-v1", "revision": "synthetic-v1",
                "device": "cpu", "dtype": "synthetic-float64", "loaded": False,
                "model": "NO_MODEL", "adapter": "synthetic-example-v1",
                "implementation": "synthetic-example-v1", "synthetic": True,
                "closed": self.closed}

    def run(self, operation, payload):
        if self.closed:
            raise RuntimeError("Fake backend is closed")
        if operation != "score":
            raise ValueError("This fixture supports score only")
        self.calls.append({"operation": operation, "payload": payload})
        objective = payload["objective"]
        if objective["kind"] == "token_margin":
            if payload.get("target") is None or payload.get("foil") is None:
                raise ValueError("Fixed target and foil required")
        elif objective["kind"] == "continuation_logprob":
            if not payload.get("continuation"):
                raise ValueError("Fixed continuation required")
        else:
            raise ValueError("Example fixture objective unsupported")

        def invented_score(text):
            digest = hashlib.sha256(text.encode()).digest()
            return -2.0 - int.from_bytes(digest[:2], "big") / 65536

        # These arithmetic rules are fixture behavior, not mechanistic findings.
        value = invented_score(payload["prompt"])
        for rule in payload.get("patch", {}).get("rules", []):
            if rule["op"] == "replace" and rule.get("source") == "clean":
                value = invented_score(payload["donor_prompt"])
            elif rule["op"] == "scale":
                value += rule["value"] - 1
            elif rule["op"] == "zero":
                value -= 1
            else:
                raise ValueError("Example fixture patch unsupported")
        return {"value": value, "objective": objective,
                "revision": "synthetic-v1", "device": "cpu",
                "dtype": "synthetic-float64",
                "input_ids": [1, 2, 3, 4, 5] if objective["kind"] == "continuation_logprob" else [1, 2, 3],
                "prompt_length": 3, "elapsed_s": 0.0,
                "diagnostics": {"synthetic": True,
                                "warning": "Invented score and token IDs; not model evidence"}}

    def close(self):
        self.closed = True


def run_demo(output, *, include_holdout=False):
    # Deferred imports keep the fake backend itself usable before integration.
    # None of these documented entry points should import/load real weights.
    from lfm3lab.research.session import ResearchSession
    from lfm3lab.research.runner import run_study
    from lfm3lab.research.report import build_report

    summaries = []
    output = Path(output)
    splits = ("discovery", "validation", "holdout") if include_holdout else (
        "discovery", "validation")
    for path in EXAMPLE_PATHS:
        study = Study.from_dict(json.loads(path.read_text()))
        backend = FakeBackend()
        with ResearchSession(backend) as session:
            # Direct score adapter: same contract as local/remote sessions.
            case = study.cases[0]
            score = session.run("score", {
                "prompt": case.prompt, "chat": False, "target": case.target,
                "foil": case.foil, "continuation": case.continuation,
                "objective": study.objective.to_dict(), "patch": {},
                "max_input_tokens": study.budget.max_input_tokens,
            })
            assert score["diagnostics"]["synthetic"] is True
            runs = [run_study(study, session, output / study.id,
                              split=split, resume=True) for split in splits]
            # Resume must reuse saved completions, not rerun successful pairs.
            before_resume = len(backend.calls)
            run_study(study, session, output / study.id,
                      split="discovery", resume=True)
            assert len(backend.calls) == before_resume, "Resume repeated scores"
            expected = sum(c.split in splits for c in study.cases) * len(study.conditions)
            assert len(backend.calls) == expected + 1, "Missing or unexpected trial calls"
        assert backend.closed
        report = build_report(output / study.id)
        for split in splits:
            data = report["splits"][split]
            assert data["status_counts"]["success"] == data["expected_trials"]
            assert data["status_counts"]["failed"] == 0
            rows = {(r["case_id"], r["condition_id"]): r for r in data["rows"]}
            for case in (c for c in study.cases if c.split == split):
                for identity_id, reference_id in study.metadata["controls"]["identity_pairs"]:
                    error = abs(rows[case.id, identity_id]["value"] - rows[case.id, reference_id]["value"])
                    assert error <= 1e-6, "Synthetic identity changed score"
        if not include_holdout:
            assert report["splits"]["holdout"]["started_trials"] == 0
        summaries.append({"study_id": study.id, "splits": list(splits),
                          "trial_calls": expected, "runs": runs,
                          "report": report})
    return {"synthetic": True, "warning": "No real model results; fixture arithmetic only",
            "studies": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Retain synthetic receipts here")
    parser.add_argument("--include-holdout", action="store_true",
                        help="Explicitly execute the visible synthetic holdout split")
    args = parser.parse_args()
    if args.output:
        result = run_demo(args.output, include_holdout=args.include_holdout)
    else:
        with tempfile.TemporaryDirectory(prefix="mechanistic-example-") as directory:
            result = run_demo(directory, include_holdout=args.include_holdout)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
