"""Read-only prepublication audit of staged files, wheel and source distribution.

This is a bounded path/content audit, not proof of secret absence or legal rights.
Publication and remote creation remain separate, owner-authorized actions.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import zipfile

ROOT_FILES = {".gitignore", ".python-version", "AGENTS.md", "README.md", "LICENSE", "NOTICE", "pyproject.toml"}
TOP_DIRS = {"src", "docs", "examples", "tests", "scripts"}
FORBIDDEN_PARTS = {"__pycache__", ".venv", ".git", "results", "evidence", "weights", "cloud", "posts", "figures", "notebooks", "data", "social", "archive"}
MACHINE = re.compile(r"/(?:Users|home)/[^\s/]+|/private" + r"/var/|/var" + r"/folders/|[A-Za-z]:\\Users\\|~/(?:lfm3-research|warm-precision)")
SECRET = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b")
FORBIDDEN_SUFFIX = {".pyc", ".safetensors", ".pt", ".pth", ".bin", ".pem", ".key", ".png", ".webp"}


def inspect(path, data):
    parts = PurePosixPath(path).parts
    assert not set(parts) & FORBIDDEN_PARTS, f"Forbidden path: {path}"
    assert PurePosixPath(path).suffix not in FORBIDDEN_SUFFIX, f"Forbidden artifact: {path}"
    assert "verification-local.json" not in parts, f"Machine-local receipt included: {path}"
    text = data.decode("utf-8")
    assert not MACHINE.search(text), f"Machine-specific reference: {path}"
    assert not SECRET.search(text), f"Possible credential: {path}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=root).decode().split("\0")
    staged = [n for n in names if n]
    assert staged, "Stage the proposed release files before auditing"
    for name in staged:
        parts = PurePosixPath(name).parts
        assert (name in ROOT_FILES or len(parts) > 1 and parts[0] in TOP_DIRS), f"Unexpected staged file: {name}"
        data = subprocess.check_output(["git", "show", f":{name}"], cwd=root)
        inspect(name, data)
        assert data == (root / name).read_bytes(), f"Staged/working mismatch: {name}"

    package = {str(p.relative_to(root / "src")) for p in (root / "src" / "lfm3lab").rglob("*.py") if "__pycache__" not in p.parts}
    resources = {f"lfm3lab/research/examples/{n}" for n in ("code_retrieval.json", "tool_continuation.json", "fake_walkthrough.py")}
    allowed_metadata = {"METADATA", "WHEEL", "entry_points.txt", "RECORD", "licenses/LICENSE", "licenses/NOTICE"}
    with zipfile.ZipFile(args.wheel) as wheel:
        wheel_names = wheel.namelist()
        expected = set(package) | resources
        actual = {n for n in wheel_names if ".dist-info/" not in n}
        assert actual == expected, f"Wheel package mismatch: {actual ^ expected}"
        for name in wheel_names:
            inspect(name, wheel.read(name))
            if ".dist-info/" in name:
                tail = name.split(".dist-info/", 1)[1]
                assert tail in allowed_metadata, f"Unexpected wheel metadata: {name}"
                if tail.startswith("licenses/"):
                    assert wheel.read(name) == (root / PurePosixPath(tail).name).read_bytes(), f"License/source mismatch: {name}"
            elif name in package:
                assert wheel.read(name) == (root / "src" / name).read_bytes(), f"Wheel/source mismatch: {name}"
            else:
                assert wheel.read(name) == (root / "examples" / "mechanistic" / PurePosixPath(name).name).read_bytes()
        metadata_name = next(n for n in wheel_names if n.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(wheel.read(metadata_name))
        assert metadata["Name"] == "glass-box" and metadata["License-Expression"] == "Apache-2.0"
        assert set(metadata.get_all("License-File", [])) == {"LICENSE", "NOTICE"}
        assert all("extra ==" in n for n in metadata.get_all("Requires-Dist", [])), "Unexpected mandatory dependencies"

    with tarfile.open(args.sdist) as sdist:
        sdist_names = []
        for member in sdist.getmembers():
            assert member.isfile(), f"Unexpected non-file sdist member: {member.name}"
            name = str(PurePosixPath(member.name).relative_to(PurePosixPath(member.name).parts[0]))
            assert name == "PKG-INFO" or name in ROOT_FILES or PurePosixPath(name).parts[0] in TOP_DIRS
            data = sdist.extractfile(member).read()
            inspect(name, data)
            if name != "PKG-INFO":
                assert data == (root / name).read_bytes(), f"Sdist/source mismatch: {name}"
            sdist_names.append(name)
        assert {"LICENSE", "NOTICE", "README.md", "pyproject.toml", "docs/library-guide.md", "docs/how-we-investigate-lfm3.html"} <= set(sdist_names)

    assert not subprocess.check_output(["git", "remote"], cwd=root).strip(), "A remote exists: review its authorization"
    print(json.dumps({"passed": True, "staged_files": len(staged), "wheel_files": len(wheel_names), "sdist_files": len(sdist_names),
                      "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                      "sdist_sha256": hashlib.sha256(args.sdist.read_bytes()).hexdigest(),
                      "machine_references_or_recognized_secrets": 0, "remote_count": 0,
                      "license": metadata["License-Expression"],
                      "limits": "Pattern-based audit; owner must confirm rights and authorize publication."}, indent=2))


if __name__ == "__main__":
    main()
