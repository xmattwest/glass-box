"""Install a wheel into fresh environments and test outside the checkout.

Needs uv and cached instrument dependencies by default. --online permits package
downloads, never model downloads. No worker rental or real inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional machine-local verification JSON")
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    offline = [] if args.online else ["--offline"]

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert any(n.endswith("/licenses/LICENSE") for n in names)
        assert any(n.endswith("/licenses/NOTICE") for n in names)
        assert len([n for n in names if n.startswith("lfm3lab/research/examples/")]) == 3
        assert not any(n.endswith((".pyc", ".safetensors", ".pt", ".pem")) for n in names)

    with tempfile.TemporaryDirectory(prefix="glass-box-wheel-") as directory:
        root = Path(directory)

        def run(command, expected=0):
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            if result.returncode != expected:
                raise RuntimeError(f"Command failed ({result.returncode}): {command}\n{result.stdout}\n{result.stderr}")
            return result.stdout

        python = root / "base" / "bin" / "python"
        run([sys.executable, "-m", "venv", "--without-pip", str(root / "base")])
        run(["uv", "pip", "install", *offline, "--python", str(python), "--no-deps", str(wheel)])
        core = json.loads(run([str(python), "-I", "-c", '''
import importlib, importlib.abc, json, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'huggingface_hub'}:
            raise AssertionError('numerical import in base install: ' + fullname)
sys.meta_path.insert(0, Deny())
import lfm3lab
from lfm3lab.research.backends import LFM3Backend
from lfm3lab.research.report import build_report
from lfm3lab.research.runner import run_study
from lfm3lab.research.worker import RemoteBackend
backend = LFM3Backend(device='cpu')
assert not backend.status()['loaded']
backend.close()
assert 'site-packages' in lfm3lab.__file__
print(json.dumps({'installed_package': lfm3lab.__file__, 'no_numerical_imports': True}))
''']))
        cli = root / "base" / "bin" / "glass-box"
        run([str(cli), "--help"])
        run([str(root / "base" / "bin" / "lfm3-research"), "--help"])
        exported = root / "studies"
        run([str(cli), "examples", "--output", str(exported)])
        for name in ("code_retrieval.json", "tool_continuation.json"):
            run([str(cli), "validate", str(exported / name)])
        # Repeat export must neither overwrite nor truncate the originals.
        before = {p.name: p.read_bytes() for p in exported.iterdir()}
        run([str(cli), "examples", "--output", str(exported)], expected=1)
        assert before == {p.name: p.read_bytes() for p in exported.iterdir()}
        demo = json.loads(run([str(python), "-I", str(exported / "fake_walkthrough.py"),
                               "--output", str(root / "synthetic")]))
        assert demo["synthetic"]
        assert sum(s["trial_calls"] for s in demo["studies"]) == 40
        for study in demo["studies"]:
            output = root / "synthetic" / study["study_id"]
            report = json.loads(run([str(cli), "report", str(output)]))
            assert report["splits"]["holdout"]["started_trials"] == 0
        # The full numerical check should give an actionable missing-extra error.
        missing = subprocess.run([str(cli), "check"], cwd=root, env=env, capture_output=True, text=True)
        assert missing.returncode == 1 and "glass-box[instruments]" in missing.stderr

        instrument_python = root / "instruments" / "bin" / "python"
        run([sys.executable, "-m", "venv", "--without-pip", str(root / "instruments")])
        run(["uv", "pip", "install", *offline, "--python", str(instrument_python), f"{wheel}[instruments]"])
        # Run the installed suite under a loader guard. It cannot fetch real weights.
        check_file = root / "installed_checks.py"
        check_file.write_text('''
import json
from unittest.mock import patch
from lfm3lab.research.checks import run_checks
if __name__ == '__main__':
    with patch('lfm3lab.model.load', side_effect=AssertionError('real model load forbidden')):
        result = run_checks()
    assert result['passed']
    print(json.dumps(result))
''')
        checks = json.loads(run([str(instrument_python), "-I", str(check_file)]))
        run([str(root / "instruments" / "bin" / "glass-box"), "--help"])
        summary = {
            "passed": True, "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "outside_checkout": True, "fresh_base_and_instruments_environments": True,
            "base": core, "base_walkthrough_trials": 40,
            "base_model_free_report": True, "export_refuses_overwrite": True,
            "no_real_model_loads": True, "installed_checks": checks,
        }
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
