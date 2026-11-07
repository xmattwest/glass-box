"""Paired, model-free descriptive reports with explicit missingness."""
from __future__ import annotations

import math
import statistics

from .schema import SPLITS, fingerprint
from .store import EvidenceStore, output_lock, trial_key, write_report_json


def _statistics(values):
    result = {"n": len(values), "mean": statistics.mean(values) if values else None,
              "sample_sd": None, "standard_error": None}
    if len(values) >= 2:
        result["sample_sd"] = statistics.stdev(values)
        result["standard_error"] = result["sample_sd"] / math.sqrt(len(values))
    return result


def _status(store, key):
    trial = store.trials.get(key)
    if trial is None:
        return "missing"
    terminal = trial["terminal"]
    if terminal is None:
        return "interrupted"
    if terminal["kind"] == "result":
        return "success"
    return "interrupted" if terminal.get("interrupted") else "failed"


def _build_report_locked(store):
    study = store.study
    warnings = [
        "Objective values and signed paired deltas are measurements, not universal improvement, generated behavior, native mechanism or practical benefit.",
        "Standard errors describe variability across paired cases, not tokens. They assume independent cases; small or dependent samples do not support population claims.",
        "Failed, interrupted and missing trials are excluded from numeric estimates but retained in denominators; missingness may bias estimates.",
        "Splits share one durable budget. Holdout access is explicit, not a security boundary.",
        "Time budgets stop new trials, not in-flight calls; interrupted trials conservatively consume their remaining time reservation.",
        "Synthetic fixture results establish software behavior only, not model findings.",
        "Control labels are protocol declarations, not automatically passed correctness/no-op tests; inspect paired control outcomes and missingness.",
    ]
    # Recognize only explicitly designated controls, never infer correctness
    # merely from a condition having no patch or a convenient name.
    controls = study.metadata.get("controls", {})
    if not isinstance(controls, dict):
        controls = {}
    declared = {c.id for c in study.conditions}
    no_op = controls.get("no_op", [])
    correctness = controls.get("correctness", [])
    no_op = no_op if isinstance(no_op, list) and all(i in declared for i in no_op) else []
    correctness = correctness if isinstance(correctness, list) and all(i in declared for i in correctness) else []
    if not no_op:
        warnings.append("No explicit no-op control declared in study.metadata.controls.no_op.")
    if not correctness:
        warnings.append("No explicit correctness control declared in study.metadata.controls.correctness.")
    secondary_pairs = study.metadata.get("secondary_comparisons", [])
    splits = {}
    for split in SPLITS:
        cases = [c for c in study.cases if c.split == split]
        rows = []
        comparisons = {}
        counts = {name: 0 for name in ("success", "failed", "interrupted", "missing")}
        for case in cases:
            baseline_key = trial_key(split, case.id, study.baseline_id)
            baseline_status = _status(store, baseline_key)
            baseline_value = store.responses[baseline_key]["value"] if baseline_status == "success" else None
            for condition in study.conditions:
                key = trial_key(split, case.id, condition.id)
                status = _status(store, key)
                counts[status] += 1
                value = store.responses[key]["value"] if status == "success" else None
                paired = status == "success" and baseline_status == "success"
                terminal = store.trials.get(key, {}).get("terminal")
                row = {"case_id": case.id, "condition_id": condition.id,
                       "status": status, "value": value, "baseline_status": baseline_status,
                       "baseline_value": baseline_value, "paired": paired,
                       "delta": value - baseline_value if paired else None,
                       "artifact": terminal.get("artifact") if terminal else None,
                       "artifact_hash": terminal.get("artifact_hash") if terminal else None,
                       "error": terminal.get("error") if terminal else None}
                rows.append(row)
        for condition in study.conditions:
            condition_rows = [r for r in rows if r["condition_id"] == condition.id]
            paired_rows = [r for r in condition_rows if r["paired"]]
            groups = {}
            for case in cases:
                group = fingerprint({k: getattr(case, k) for k in (
                    "prompt", "corrupt_prompt", "target", "foil", "continuation")})
                groups.setdefault(group, []).append(case.id)
            repeatability = []
            for group, ids in groups.items():
                if len(ids) < 2:
                    continue
                repeats = [r["delta"] for r in paired_rows if r["case_id"] in ids]
                repeatability.append({"input_fingerprint": group, "case_ids": ids,
                                      "expected_repeats": len(ids), "paired_delta": _statistics(repeats)})
            comparison = {"baseline_id": study.baseline_id, "expected_cases": len(cases),
                          "paired_cases": len(paired_rows), "unpaired_cases": len(cases) - len(paired_rows),
                          "condition_status_counts": {name: sum(r["status"] == name for r in condition_rows) for name in counts},
                          "baseline_status_counts": {name: sum(r["baseline_status"] == name for r in condition_rows) for name in counts},
                          "delta": _statistics([r["delta"] for r in paired_rows]),
                          "repeatability": repeatability}
            if repeatability:
                comparison["uncertainty_warning"] = "Repeated identical inputs are not independent cases; the pooled standard error is descriptive only."
            elif len(paired_rows) < 2:
                comparison["uncertainty_warning"] = "Fewer than two paired cases: no variance or uncertainty estimate."
            comparisons[condition.id] = comparison
        by_case = {(row["case_id"], row["condition_id"]): row for row in rows}
        secondary = {}
        for left_id, right_id in secondary_pairs:
            paired_rows = []
            case_rows = []
            left_counts = {name: 0 for name in counts}
            right_counts = {name: 0 for name in counts}
            for case in cases:
                left = by_case[(case.id, left_id)]
                right = by_case[(case.id, right_id)]
                left_counts[left["status"]] += 1
                right_counts[right["status"]] += 1
                paired = left["status"] == right["status"] == "success"
                delta = left["value"] - right["value"] if paired else None
                if paired:
                    paired_rows.append(delta)
                case_rows.append({"case_id": case.id, "left_status": left["status"],
                                  "right_status": right["status"], "left_value": left["value"],
                                  "right_value": right["value"], "paired": paired, "delta": delta,
                                  "left_artifact": left["artifact"], "left_artifact_hash": left["artifact_hash"],
                                  "right_artifact": right["artifact"], "right_artifact_hash": right["artifact_hash"]})
            key = f"{left_id} - {right_id}"
            secondary[key] = {"left_id": left_id, "right_id": right_id,
                              "expected_cases": len(cases), "paired_cases": len(paired_rows),
                              "unpaired_cases": len(cases) - len(paired_rows),
                              "left_status_counts": left_counts, "right_status_counts": right_counts,
                              "delta": _statistics(paired_rows), "case_rows": case_rows}
            if len(paired_rows) < 2:
                secondary[key]["uncertainty_warning"] = "Fewer than two paired cases: no variance or uncertainty estimate."
        executed = sum(name != "missing" for name in [r["status"] for r in rows])
        splits[split] = {"expected_cases": len(cases), "expected_trials": len(cases) * len(study.conditions),
                          "started_trials": executed, "status_counts": counts,
                          "comparisons": comparisons, "secondary_comparisons": secondary, "rows": rows}
    report = {"format_version": 1, "study_id": study.id, "question": study.question,
              "prediction": study.prediction, "objective": study.objective.to_dict(),
              "baseline_id": study.baseline_id, "seed": study.seed,
              "study_metadata": study.metadata, "backend_identity": store.protocol["backend_identity"],
              "budget": study.budget.to_dict(), "budget_used": store.budget_used(),
              "splits": splits, "controls": {"no_op": no_op, "correctness": correctness},
              "limitations": warnings,
              "provenance": {"study_fingerprint": study.fingerprint,
                             "snapshot_hash": store.protocol["snapshot_hash"],
                             "journal_head": store.events[-1]["event_hash"] if store.events else None,
                             "journal_events": len(store.events),
                             "protocol": "protocol.json", "journal": "journal.jsonl",
                             "verification": "Protocol, journal hash chain and all completed raw artifact hashes verified",
                             "output": str(store.path.resolve())}}
    write_report_json(store.path / "report.json", report)
    lines = [f"# Study: {study.id}", "", f"Question: {study.question}", "",
             f"Prediction: {study.prediction}", "", f"Objective: `{study.objective.kind}` ({study.objective.reduction}); baseline: `{study.baseline_id}`.", "",
             "Signed deltas are condition minus explicit baseline, not improvement claims.", "",
             "| Split | Condition | Paired / expected cases | Mean delta | Case-level SE | Failed / interrupted / missing |",
             "|---|---|---:|---:|---:|---:|"]
    for split, data in splits.items():
        for condition, comparison in data["comparisons"].items():
            if not data["expected_cases"]:
                continue
            stats = comparison["delta"]
            display = lambda x: "not estimable" if x is None else f"{x:.6g}"
            counts = comparison["condition_status_counts"]
            lines.append(f"| {split} | {condition} | {stats['n']} / {comparison['expected_cases']} | {display(stats['mean'])} | {display(stats['standard_error'])} | {counts['failed']} / {counts['interrupted']} / {counts['missing']} |")
    if secondary_pairs:
        lines += ["", "## Predeclared paired contrasts", "",
                  "Signed delta is left condition minus right condition. Missing or failed pairs have no numeric effect.", "",
                  "| Split | Left - right | Paired / expected cases | Mean delta | Case-level SE |",
                  "|---|---|---:|---:|---:|"]
        for split, data in splits.items():
            if not data["expected_cases"]:
                continue
            for key, comparison in data["secondary_comparisons"].items():
                stats = comparison["delta"]
                lines.append(f"| {split} | {key} | {stats['n']} / {comparison['expected_cases']} | {display(stats['mean'])} | {display(stats['standard_error'])} |")
    lines += ["", "## Limitations", "", *[f"- {w}" for w in warnings], "", "## Provenance", "",
              f"- Protocol fingerprint: `{study.fingerprint}`", f"- Snapshot hash: `{store.protocol['snapshot_hash']}`",
              f"- Journal head: `{report['provenance']['journal_head']}`",
              "- See `report.json` for per-case raw artifact paths, hashes, missingness and repeatability.", ""]
    # Derived Markdown is replaceable; immutable source receipts are untouched.
    markdown = store.path / "report.md"
    temporary = markdown.with_name("report.md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(markdown)
    return report


def build_report(output):
    """Verify evidence and regenerate both reports without opening any model."""
    with output_lock(output) as path:
        return _build_report_locked(EvidenceStore(path).load())
