"""Consumer first-use regressions. Synthetic evidence only; no real model."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]


def example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/mechanistic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FirstUseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = example("fake_walkthrough")
        cls.audit = example("audit_controls")
        cls.temp = tempfile.TemporaryDirectory(prefix="first-use-test-")
        cls.result = cls.demo.run_demo(cls.temp.name)
        cls.report = cls.result["studies"][0]["report"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_v2_comparisons_are_operative(self):
        self.assertEqual(self.report["study_id"], "code-retrieval-mechanics-v2")
        self.assertTrue(self.report["splits"]["discovery"]["secondary_comparisons"])
        self.assertNotIn("secondary_comparisons", self.report["study_metadata"]["controls"])

    def test_identity_uses_declared_reference_not_primary_baseline(self):
        result = self.audit.audit_report(self.report, tolerance=1e-6)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["check_counts"]["passed"], 8)
        self.assertEqual(result["max_absolute_error"], 0)
        corrupt = [c for c in result["checks"] if c["identity_id"] == "corrupt_identity"]
        self.assertTrue(all(c["reference_id"] == "corrupt" for c in corrupt))

    def test_violation_and_missing_never_pass(self):
        report = copy.deepcopy(self.report)
        row = next(r for r in report["splits"]["discovery"]["rows"] if r["condition_id"] == "identity")
        row["value"] += 0.5
        self.assertEqual(self.audit.audit_report(report, tolerance=0.125)["status"], "failed")
        row["status"] = "failed"
        row["value"] = None
        self.assertEqual(self.audit.audit_report(report, tolerance=0.125)["status"], "incomplete")
        holdout = self.audit.audit_report(self.report, tolerance=1e-6, splits=["holdout"])
        self.assertEqual(holdout["status"], "incomplete")
        self.assertEqual(holdout["check_counts"]["passed"], 0)

    def test_empty_split_and_undeclared_pairs_do_not_pass(self):
        report = copy.deepcopy(self.report)
        report["splits"]["discovery"]["rows"] = []
        self.assertEqual(self.audit.audit_report(report, tolerance=0, splits=["discovery"])["status"], "incomplete")
        report["study_metadata"]["controls"]["identity_pairs"] = []
        with self.assertRaisesRegex(ValueError, "No controls.identity_pairs"):
            self.audit.audit_report(report, tolerance=0)

    def test_invalid_audit_requests(self):
        for tolerance in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                self.audit.audit_report(self.report, tolerance=tolerance)
        for pairs in ([["missing", "clean"]], [["clean", "clean"]],
                      [["identity", "clean"], ["identity", "clean"]]):
            report = copy.deepcopy(self.report)
            report["study_metadata"]["controls"]["identity_pairs"] = pairs
            with self.assertRaises(ValueError):
                self.audit.audit_report(report, tolerance=0)

    def test_compact_walkthrough_retains_and_counts_evidence(self):
        with tempfile.TemporaryDirectory(prefix="compact-first-use-") as directory:
            command = [sys.executable, str(ROOT / "examples/mechanistic/fake_walkthrough.py"),
                       "--output", directory, "--summary"]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            data = json.loads(result.stdout)
            self.assertTrue(data["receipts_retained"])
            self.assertLess(len(result.stdout), 6000)
            self.assertEqual(sum(s["total_score_calls"] for s in data["studies"]), 42)
            self.assertEqual(sum(s["resume_score_calls"] for s in data["studies"]), 0)
            self.assertTrue(all((Path(s["output"]) / "protocol.json").exists() for s in data["studies"]))

    def test_cli_validation_does_not_claim_model_readiness(self):
        from lfm3lab.research.cli import main
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = main(["validate", str(ROOT / "examples/mechanistic/code_retrieval.json")])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stream.getvalue())["validation"], {
            "level": "schema", "native_coordinates": "not_checked",
            "tokenization": "not_checked", "model_loaded": False})

    def test_audit_cli_verifies_and_preserves_raw_evidence(self):
        folder = Path(self.temp.name) / self.report["study_id"]
        paths = [folder / "protocol.json", folder / "journal.jsonl", *sorted((folder / "raw").glob("*.json"))]
        before = {p: p.read_bytes() for p in paths}
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = self.audit.main([str(folder), "--tolerance", "0.000001"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stream.getvalue())["status"], "passed")
        self.assertEqual(before, {p: p.read_bytes() for p in paths})


if __name__ == "__main__":
    unittest.main()
