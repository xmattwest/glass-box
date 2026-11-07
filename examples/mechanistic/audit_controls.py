"""Audit declared identity pairs from verified receipts, without model calls.

This consumer helper is not an automatic correctness test. Fix the absolute
score tolerance before examining results; do not infer it from prose metadata.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from lfm3lab.research import build_report


def audit_report(report, *, tolerance, splits=("discovery", "validation")):
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    pairs = report["study_metadata"].get("controls", {}).get("identity_pairs", [])
    if not pairs:
        raise ValueError("No controls.identity_pairs declared; no identity audit is possible")
    condition_ids = set(next(iter(report["splits"].values()))["comparisons"])
    seen = set()
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(x, str) or x not in condition_ids for x in pair)
                or pair[0] == pair[1] or tuple(pair) in seen):
            raise ValueError("identity_pairs must be unique [identity, reference] condition pairs")
        seen.add(tuple(pair))
    if not splits or len(set(splits)) != len(splits) or any(s not in report["splits"] for s in splits):
        raise ValueError("Select distinct existing splits")
    checks = []
    for split in splits:
        data = report["splits"][split]
        rows = {(r["case_id"], r["condition_id"]): r for r in data["rows"]}
        for case_id in sorted({r["case_id"] for r in data["rows"]}):
            for identity, reference in pairs:
                left, right = rows[case_id, identity], rows[case_id, reference]
                available = left["status"] == right["status"] == "success"
                error = abs(left["value"] - right["value"]) if available else None
                if error is not None and not math.isfinite(error):
                    raise ValueError("Nonfinite identity error")
                checks.append({
                    "split": split, "case_id": case_id,
                    "identity_id": identity, "reference_id": reference,
                    "identity_status": left["status"], "reference_status": right["status"],
                    "absolute_error": error,
                    "status": "unavailable" if not available else
                              ("passed" if error <= tolerance else "failed"),
                })
    counts = {status: sum(c["status"] == status for c in checks)
              for status in ("passed", "failed", "unavailable")}
    status = "failed" if counts["failed"] else (
        "incomplete" if counts["unavailable"] or not checks else "passed")
    errors = [c["absolute_error"] for c in checks if c["absolute_error"] is not None]
    return {
        "study_id": report["study_id"], "status": status, "absolute_tolerance": tolerance,
        "scope": "Declared identity pairs only; not generated correctness or study validity",
        "backend_identity": report["backend_identity"], "provenance": report["provenance"],
        "splits": list(splits), "check_counts": counts,
        "max_absolute_error": max(errors) if errors else None, "checks": checks,
        "trial_status_counts": {s: report["splits"][s]["status_counts"] for s in splits},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Existing evidence directory")
    parser.add_argument("--tolerance", type=float, required=True,
                        help="Predetermined absolute score tolerance, in objective units")
    parser.add_argument("--split", choices=("discovery", "validation", "holdout"),
                        action="append", help="Repeat to select splits; default discovery + validation")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.tolerance) or args.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        result = audit_report(build_report(args.output), tolerance=args.tolerance,
                              splits=args.split or ("discovery", "validation"))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": {"type": type(exc).__name__,
                                                "message": str(exc)}}, allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return {"passed": 0, "failed": 1, "incomplete": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
