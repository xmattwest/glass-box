"""Deterministic, bounded study execution, with no model imports."""
from __future__ import annotations

import math
import time

from .schema import SPLITS, Study, canonical_json
from .store import EvidenceStore, backend_identity, output_lock, trial_key, write_new_json


def score_payload(study, case, condition):
    payload = {"prompt": case.prompt if condition.input_variant == "clean" else case.corrupt_prompt,
               "chat": False, "target": case.target, "foil": case.foil,
               "continuation": case.continuation, "objective": study.objective.to_dict(),
               "patch": condition.patch, "max_input_tokens": study.budget.max_input_tokens}
    if any(rule.get("source") == "clean" for rule in condition.patch.get("rules", [])):
        payload["donor_prompt"] = case.prompt
    return payload


def _validate_response(response, study, identity):
    if not isinstance(response, dict):
        raise ValueError("Score response must be a JSON object")
    canonical_json(response)
    value = response.get("value")
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Score response needs a finite numeric value")
    if response.get("objective") != study.objective.to_dict():
        raise ValueError("Score response objective differs from protocol")
    for key in ("revision", "device", "dtype"):
        if response.get(key) != identity[key]:
            raise ValueError(f"Score response {key} differs from backend identity")
    if not isinstance(response.get("input_ids"), list) or any(type(i) is not int or i < 0 for i in response["input_ids"]):
        raise ValueError("Score response needs input token IDs")
    length = response.get("prompt_length")
    if type(length) is not int or not 1 <= length <= len(response["input_ids"]):
        raise ValueError("Score response prompt_length must index a nonempty prompt in input_ids")
    if study.objective.kind != "continuation_logprob" and length != len(response["input_ids"]):
        raise ValueError("Token score response prompt_length must match input_ids")
    if len(response["input_ids"]) > study.budget.max_input_tokens:
        raise ValueError("Backend exceeded max_input_tokens")
    if type(response.get("elapsed_s")) not in (int, float) or response["elapsed_s"] < 0:
        raise ValueError("Score response needs nonnegative elapsed_s")


def run_study(study, session, output, *, split="discovery", resume=True):
    """Never retries a started pair; budgets apply across all explicit splits.

    Time limits prevent starting another call, not interrupting an in-flight
    backend operation. Interrupted calls consume their remaining reservation.
    """
    study = Study.from_dict(study)  # detached snapshot, even if caller mutates it
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    with output_lock(output) as path:
        identity = backend_identity(session.status())
        store = EvidenceStore(path)
        store.initialize(study, identity, resume=resume)
        store.recover_interrupted()
        stop_reason = "complete"
        for case in study.cases:
            if case.split != split:
                continue
            for condition in study.conditions:
                key = trial_key(split, case.id, condition.id)
                if key in store.trials:
                    continue
                used = store.budget_used()
                if used["attempts"] >= study.budget.max_trials:
                    stop_reason = "max_trials"
                    break
                remaining = study.budget.max_seconds - used["elapsed_s"]
                if remaining <= 0:
                    stop_reason = "max_seconds"
                    break
                payload = score_payload(study, case, condition)
                intent = store.append("intent", key, split=split, case_id=case.id,
                                      condition_id=condition.id, payload=payload,
                                      reserved_seconds=remaining)
                started = time.monotonic()
                digest = None
                artifact = f"raw/{intent['seq']:08d}.json"
                try:
                    # Pass a JSON copy: a backend must not mutate the intent.
                    import json
                    response = session.score(**json.loads(canonical_json(payload)))
                    # Preserve JSON-safe invalid responses too, as failure
                    # diagnostics; never count them as successful measurements.
                    digest = write_new_json(path / artifact, {"key": key, "payload": payload, "response": response})
                    _validate_response(response, study, identity)
                except Exception as exc:
                    store.append("failure", key, elapsed_s=time.monotonic() - started,
                                 error_type=type(exc).__name__, error=str(exc), interrupted=False,
                                 **({"artifact": artifact, "artifact_hash": digest} if digest else {}))
                else:
                    store.append("result", key, elapsed_s=time.monotonic() - started,
                                 artifact=artifact, artifact_hash=digest)
            if stop_reason != "complete":
                break
        from .report import _build_report_locked
        report = _build_report_locked(store)
        report["run"] = {"requested_split": split, "stop_reason": stop_reason,
                         "budget_used": store.budget_used(), "resume": resume}
        return report
