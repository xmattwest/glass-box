"""CPU-only evidence/runner checks. Run with python -m ...checks_runner."""
from __future__ import annotations

import copy
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from .report import build_report
from .runner import run_study, score_payload
from .schema import Budget, Case, Condition, Objective, Study
from .store import EvidenceError, EvidenceStore, OutputLocked, backend_identity, output_lock, trial_key


class FakeSession:
    def __init__(self, *, failing=False):
        self.calls = []
        self.failing = failing
        self.identity = {"revision": "fixture-r1", "device": "cpu", "dtype": "float64",
                         "adapter": "fixture", "implementation": "fixture-v1"}
        self.volatile = 0

    def status(self):
        self.volatile += 1
        return {**self.identity, "loaded": bool(self.volatile % 2), "uptime": self.volatile,
                "host": "localhost", "url": f"http://host-{self.volatile}"}

    def score(self, **payload):
        self.calls.append(copy.deepcopy(payload))
        if self.failing and payload["patch"]:
            raise RuntimeError("fixture failure")
        base = float(payload["prompt"].split()[-1])
        return {"value": base + (2 if payload["patch"] else 0),
                "objective": payload["objective"], **self.identity,
                "input_ids": [1, 2], "prompt_length": 2, "elapsed_s": 0.001}


def fixture(*, max_trials=128, max_seconds=600):
    return Study(id="fixture", question="Does the fixture preserve paired measurements?",
                 prediction="The programmed offset is two, not a scientific finding.",
                 cases=[Case("a", "case 1", target=1, foil=2, corrupt_prompt="corrupt 8"),
                        Case("b", "case 3", target=1, foil=2, corrupt_prompt="corrupt 9"),
                        Case("v", "case 4", target=1, foil=2, corrupt_prompt="corrupt 7", split="validation"),
                        Case("h", "case 5", target=1, foil=2, corrupt_prompt="corrupt 6", split="holdout")],
                 conditions=[Condition("clean"), Condition("shift", patch={"rules": [{"unit": "layer:0", "op": "zero"}]})],
                 budget=Budget(max_trials=max_trials, max_seconds=max_seconds))


def _lock_child(path, connection):
    try:
        with output_lock(path):
            connection.send("acquired")
    except OutputLocked:
        connection.send("locked")
    finally:
        connection.close()


class RunnerChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lfm3-evidence-check-")
        self.output = Path(self.temporary.name) / "run"
        self.study = fixture()
        self.session = FakeSession()

    def tearDown(self):
        self.temporary.cleanup()

    def test_resume_split_pairing_and_provenance(self):
        report = run_study(self.study, self.session, self.output)
        self.assertEqual(len(self.session.calls), 4)
        self.assertEqual(report["splits"]["holdout"]["started_trials"], 0)
        self.assertEqual(report["splits"]["validation"]["started_trials"], 0)
        stats = report["splits"]["discovery"]["comparisons"]["shift"]["delta"]
        self.assertEqual(stats, {"n": 2, "mean": 2.0, "sample_sd": 0.0, "standard_error": 0.0})
        journal = (self.output / "journal.jsonl").read_bytes()
        raw = {p.name: p.read_bytes() for p in (self.output / "raw").iterdir()}
        run_study(self.study, self.session, self.output)
        self.assertEqual(len(self.session.calls), 4)
        self.assertEqual((self.output / "journal.jsonl").read_bytes(), journal)
        self.assertEqual({p.name: p.read_bytes() for p in (self.output / "raw").iterdir()}, raw)
        run_study(self.study, self.session, self.output, split="validation")
        self.assertEqual(len(self.session.calls), 6)
        final = run_study(self.study, self.session, self.output, split="holdout")
        self.assertEqual(len(self.session.calls), 8)
        self.assertEqual(final["budget_used"]["attempts"], 8)
        self.assertEqual(final["provenance"]["study_fingerprint"], self.study.fingerprint)
        self.assertTrue((self.output / "report.md").exists())
        first = build_report(self.output)
        self.assertEqual(first, build_report(self.output))

    def test_incompatible_study_backend_and_no_resume(self):
        run_study(self.study, self.session, self.output)
        journal = (self.output / "journal.jsonl").read_bytes()
        changed = Study.from_dict(self.study)
        changed.prediction = "Changed after execution"
        with self.assertRaises(EvidenceError):
            run_study(changed, self.session, self.output)
        for key in ("revision", "device", "dtype", "adapter", "implementation"):
            other = FakeSession()
            other.identity[key] += "-different"
            with self.assertRaises(EvidenceError):
                run_study(self.study, other, self.output)
            self.assertEqual(other.calls, [])
        with self.assertRaises(EvidenceError):
            run_study(self.study, self.session, self.output, resume=False)
        self.assertEqual((self.output / "journal.jsonl").read_bytes(), journal)

    def test_failed_attempts_count_and_are_not_retried(self):
        study = fixture(max_trials=3)
        session = FakeSession(failing=True)
        report = run_study(study, session, self.output)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(report["run"]["stop_reason"], "max_trials")
        counts = report["splits"]["discovery"]["status_counts"]
        self.assertEqual(counts, {"success": 2, "failed": 1, "interrupted": 0, "missing": 1})
        comparison = report["splits"]["discovery"]["comparisons"]["shift"]
        self.assertEqual(comparison["paired_cases"], 0)
        self.assertIsNone(comparison["delta"]["mean"])
        self.assertEqual(comparison["expected_cases"], 2)
        run_study(study, session, self.output, split="holdout")
        run_study(study, session, self.output)
        self.assertEqual(len(session.calls), 3)
        events = [json.loads(line) for line in (self.output / "journal.jsonl").read_text().splitlines()]
        failure = [e for e in events if e["kind"] == "failure"][0]
        self.assertEqual(failure["error_type"], "RuntimeError")
        self.assertEqual(failure["error"], "fixture failure")

    def test_missing_baseline_not_fabricated(self):
        study = fixture()
        study.conditions.reverse()
        study.budget.max_trials = 1
        report = run_study(study, self.session, self.output)
        row = next(r for r in report["splits"]["discovery"]["rows"] if r["case_id"] == "a" and r["condition_id"] == "shift")
        self.assertEqual(row["value"], 3)
        self.assertIsNone(row["baseline_value"])
        self.assertIsNone(row["delta"])
        self.assertFalse(row["paired"])

    def test_predeclared_secondary_pairs_and_failures(self):
        study = fixture()
        study.conditions = [Condition("clean"), Condition("corrupt", input_variant="corrupt"),
                            Condition("restore", input_variant="corrupt", patch={"rules": [
                                {"unit": "layer:0", "op": "zero"}]})]
        study.metadata["secondary_comparisons"] = [["clean", "corrupt"],
                                                    ["restore", "corrupt"]]
        report = run_study(study, self.session, self.output)
        pairs = report["splits"]["discovery"]["secondary_comparisons"]
        self.assertEqual(pairs["clean - corrupt"]["delta"]["mean"], -6.5)
        restore = pairs["restore - corrupt"]
        self.assertEqual(restore["delta"], {"n": 2, "mean": 2, "sample_sd": 0, "standard_error": 0})
        self.assertEqual(restore["case_rows"][0]["right_artifact_hash"],
                         report["splits"]["discovery"]["rows"][1]["artifact_hash"])
        self.assertEqual(report["splits"]["holdout"]["secondary_comparisons"]["restore - corrupt"]["paired_cases"], 0)
        self.assertEqual(build_report(self.output)["splits"]["discovery"]["secondary_comparisons"], pairs)
        self.assertIn("Predeclared paired contrasts", (self.output / "report.md").read_text())

        with self.assertRaises(ValueError):
            invalid = Study.from_dict(study)
            invalid.metadata["secondary_comparisons"] = [["restore", "absent"]]
            invalid.validate()
        with self.assertRaises(ValueError):
            invalid = Study.from_dict(study)
            invalid.metadata["secondary_comparisons"] *= 2
            invalid.validate()

    def test_secondary_pairs_do_not_impute_failed_or_missing_scores(self):
        study = fixture(max_trials=5)
        study.conditions = [Condition("clean"), Condition("corrupt", input_variant="corrupt"),
                            Condition("restore", input_variant="corrupt", patch={"rules": [
                                {"unit": "layer:0", "op": "zero"}]})]
        study.metadata["secondary_comparisons"] = [["restore", "corrupt"]]
        report = run_study(study, FakeSession(failing=True), self.output)
        comparison = report["splits"]["discovery"]["secondary_comparisons"]["restore - corrupt"]
        self.assertEqual(comparison["paired_cases"], 0)
        self.assertEqual(comparison["delta"]["mean"], None)
        self.assertEqual(comparison["left_status_counts"],
                         {"success": 0, "failed": 1, "interrupted": 0, "missing": 1})
        self.assertEqual([r["delta"] for r in comparison["case_rows"]], [None, None])

    def test_interrupted_and_partial_report(self):
        with output_lock(self.output):
            store = EvidenceStore(self.output)
            store.initialize(self.study, backend_identity(self.session.status()), resume=True)
            case, condition = self.study.cases[0], self.study.conditions[0]
            key = trial_key(case.split, case.id, condition.id)
            store.append("intent", key, split=case.split, case_id=case.id, condition_id=condition.id,
                         payload=score_payload(self.study, case, condition), reserved_seconds=600)
        report = build_report(self.output)
        self.assertEqual(report["splits"]["discovery"]["status_counts"],
                         {"success": 0, "failed": 0, "interrupted": 1, "missing": 3})
        self.assertEqual(report["budget_used"], {"attempts": 1, "elapsed_s": 600})
        resumed = run_study(self.study, self.session, self.output)
        self.assertEqual(self.session.calls, [])
        self.assertEqual(resumed["run"]["stop_reason"], "max_seconds")
        with output_lock(self.output):
            store = EvidenceStore(self.output).load()
            self.assertEqual(store.trials[key]["terminal"]["error_type"], "InterruptedAttempt")
        count = (self.output / "journal.jsonl").read_bytes()
        run_study(self.study, self.session, self.output)
        self.assertEqual((self.output / "journal.jsonl").read_bytes(), count)

    def test_seconds_budget_counts_failure(self):
        # No sleeping: fake a slow call's observed monotonic duration.
        from unittest.mock import patch
        study = fixture(max_seconds=1)
        session = FakeSession(failing=True)
        study.conditions.reverse()
        with patch("lfm3lab.research.runner.time.monotonic", side_effect=[0.0, 2.0]):
            report = run_study(study, session, self.output)
        self.assertEqual(report["budget_used"]["elapsed_s"], 2)
        self.assertEqual(report["run"]["stop_reason"], "max_seconds")
        run_study(study, session, self.output)
        self.assertEqual(len(session.calls), 1)

    def test_raw_protocol_and_journal_corruption_refused(self):
        run_study(self.study, self.session, self.output)
        paths = [self.output / "protocol.json", self.output / "journal.jsonl",
                 next((self.output / "raw").iterdir())]
        for path in paths:
            original = path.read_bytes()
            damaged = original.replace(b'"seq":0', b'"seq":99', 1) if path.name == "journal.jsonl" else original.replace(b"fixture", b"tampered", 1)
            self.assertNotEqual(damaged, original)
            path.write_bytes(damaged)
            with self.assertRaises(EvidenceError):
                build_report(self.output)
            with self.assertRaises(EvidenceError):
                run_study(self.study, self.session, self.output)
            path.write_bytes(original)
        journal = paths[1]
        original = journal.read_bytes()
        journal.write_bytes(original[:-1])
        with self.assertRaises(EvidenceError):
            build_report(self.output)
        journal.write_bytes(original)
        self.assertEqual(len(self.session.calls), 4)

    def test_lock_safety_and_release(self):
        with output_lock(self.output):
            with self.assertRaises(OutputLocked):
                run_study(self.study, self.session, self.output)
            with self.assertRaises(OutputLocked):
                build_report(self.output)
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(target=_lock_child, args=(str(self.output), child))
            process.start()
            self.assertTrue(parent.poll(10), "Lock subprocess did not answer")
            self.assertEqual(parent.recv(), "locked")
            process.join(10)
            self.assertEqual(process.exitcode, 0)
            parent.close()
            child.close()
        run_study(self.study, self.session, self.output)
        self.assertEqual(len(self.session.calls), 4)

    def test_corrupt_payload_fixed_pair_and_donor(self):
        condition = Condition("restore", input_variant="corrupt", patch={"rules": [
            {"unit": "layer:0", "op": "replace", "source": "clean"}]})
        self.study.conditions.append(condition)
        run_study(self.study, self.session, self.output)
        payload = self.session.calls[2]
        self.assertEqual(payload["prompt"], "corrupt 8")
        self.assertEqual(payload["donor_prompt"], "case 1")
        self.assertEqual((payload["target"], payload["foil"]), (1, 2))
        self.assertEqual(payload["max_input_tokens"], 256)
        self.assertEqual(payload["objective"], self.study.objective.to_dict())

    def test_repeatability_and_no_uncertainty_for_one_case(self):
        study = fixture()
        study.cases[1].prompt = study.cases[0].prompt
        study.cases[1].corrupt_prompt = study.cases[0].corrupt_prompt
        report = run_study(study, self.session, self.output)
        comparison = report["splits"]["discovery"]["comparisons"]["shift"]
        self.assertEqual(comparison["repeatability"][0]["paired_delta"]["n"], 2)
        self.assertIn("not independent", comparison["uncertainty_warning"])
        report = run_study(study, self.session, self.output, split="validation")
        stats = report["splits"]["validation"]["comparisons"]["shift"]["delta"]
        self.assertIsNone(stats["standard_error"])
        self.assertIsNone(stats["sample_sd"])
        self.assertTrue(any("no-op" in w for w in report["limitations"]))

    def test_empty_partial_report_and_nonempty_output_refusal(self):
        with output_lock(self.output):
            EvidenceStore(self.output).initialize(self.study, backend_identity(self.session.status()), resume=True)
        report = build_report(self.output)
        self.assertEqual(report["splits"]["discovery"]["status_counts"]["missing"], 4)
        elsewhere = Path(self.temporary.name) / "unrelated"
        elsewhere.mkdir()
        (elsewhere / "user-work.txt").write_text("preserve")
        with self.assertRaises(EvidenceError):
            run_study(self.study, self.session, elsewhere)
        self.assertEqual((elsewhere / "user-work.txt").read_text(), "preserve")

    def test_real_call_interruption_never_retried(self):
        class Interrupted(FakeSession):
            def score(inner, **payload):
                inner.calls.append(payload)
                raise KeyboardInterrupt("fixture interruption")
        interrupted = Interrupted()
        with self.assertRaises(KeyboardInterrupt):
            run_study(self.study, interrupted, self.output)
        report = build_report(self.output)
        self.assertEqual(report["budget_used"]["attempts"], 1)
        self.assertEqual(report["splits"]["discovery"]["status_counts"]["interrupted"], 1)
        run_study(self.study, self.session, self.output)
        self.assertEqual(self.session.calls, [])

    def test_continuation_teacher_forced_receipts(self):
        self.study.objective = Objective("continuation_logprob", "mean")
        for case in self.study.cases:
            case.continuation = "a fixed continuation"
        class Continuation(FakeSession):
            def score(inner, **payload):
                response = super().score(**payload)
                response["input_ids"] += [3, 4]
                return response
        report = run_study(self.study, Continuation(), self.output)
        self.assertEqual(report["splits"]["discovery"]["status_counts"]["success"], 4)
        self.assertEqual(report["objective"], {"kind": "continuation_logprob", "reduction": "mean"})

    def test_invalid_score_is_journaled_not_paired(self):
        class Invalid(FakeSession):
            def score(inner, **payload):
                response = super().score(**payload)
                response["objective"] = {"kind": "token_logprob", "reduction": "sum"}
                return response
        report = run_study(self.study, Invalid(), self.output)
        self.assertEqual(report["budget_used"]["attempts"], 4)
        self.assertEqual(report["splits"]["discovery"]["status_counts"]["failed"], 4)
        self.assertEqual(report["splits"]["discovery"]["comparisons"]["shift"]["paired_cases"], 0)
        row = report["splits"]["discovery"]["rows"][0]
        self.assertTrue((self.output / row["artifact"]).exists())
        self.assertIsNone(row["value"])
        (self.output / row["artifact"]).write_text("corrupt failure receipt")
        with self.assertRaises(EvidenceError):
            build_report(self.output)


def run_checks():
    import io
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RunnerChecks))
    return {"ok": result.wasSuccessful(), "checks": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "details": stream.getvalue(), "real_inference": False}


if __name__ == "__main__":
    result = run_checks()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
