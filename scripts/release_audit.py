"""Read-only publication audit of the complete Git index, wheel and sdist.

Stage every intended publication change, rebuild both distributions, then run:
    python scripts/release_audit.py WHEEL SDIST
After authorized remote creation, add:
    --expected-remote-url https://github.com/xmattwest/glass-box.git

The index must match working files. Only check.yml and pages.yml are reviewed
GitHub workflows; other hidden configuration is excluded. Git identity and
commit history are never inspected or rewritten by this command. This bounded
path/content audit is not proof of secret absence or redistribution rights.
"""
from __future__ import annotations

import argparse
import email
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tomllib
import zipfile

ROOT_FILES = {".gitignore", ".python-version", "AGENTS.md", "README.md", "LICENSE", "NOTICE", "pyproject.toml", "llms.txt", "llms-full.txt"}
TOP_DIRS = {"src", "docs", "examples", "tests", "scripts"}
WORKFLOWS = {".github/workflows/check.yml", ".github/workflows/pages.yml"}
AUTHORIZED_REMOTE_URLS = {
    "https://github.com/xmattwest/glass-box",
    "https://github.com/xmattwest/glass-box.git",
    "git@github.com:xmattwest/glass-box.git",
    "ssh://git@github.com/xmattwest/glass-box.git",
}
FORBIDDEN_PARTS = {"__pycache__", ".venv", ".git", "results", "evidence", "weights", "cloud", "posts", "figures", "notebooks", "data", "social", "archive", "raw", "studio", ".site"}
FORBIDDEN_NAMES = {"verification-local.json", "journal.jsonl", "lab-log.md", "lab_log.md", "lablog.md"}
FORBIDDEN_SUFFIX = {".pyc", ".safetensors", ".pt", ".pth", ".bin", ".pem", ".key", ".png", ".webp", ".ipynb", ".ckpt", ".npy", ".npz"}
MACHINE = re.compile(
    r"/(?:Users|home)/(?P<unix>[^/\s\"'<>]+)"
    r"|[A-Za-z]:\\+(?:Users|Documents and Settings)\\+(?P<windows>[^\\\s\"'<>]+)"
    r"|/private" + r"/var/|/var" + r"/folders/|~/(?:lfm3-research|warm-precision)"
)
SAMPLE_USERS = {"user", "username", "example", "your-user", "${USER}", "{user}", "{username}"}
SECRET = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED |DSA )?PRIVATE KEY-----"
    r"|\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})\b"
    r"|\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\b[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_+/=-]{20,}[\"']",
    re.IGNORECASE,
)
URL_CREDENTIAL = re.compile(r"https?://(?P<user>[^\s/:<>\"']+):(?P<password>[^\s/@<>\"']+)@", re.IGNORECASE)
SAMPLE_CREDENTIALS = {("user", "pass"), ("user", "password"), ("username", "password")}
STALE_URL = re.compile(r"(?:github\.com/|git@github\.com:)[^/\s]+/(?:lfm3-research|lfm3-glassbox)(?=[/.#?\s\"')]|$)", re.IGNORECASE)
REQUIRED_SDIST = {"LICENSE", "NOTICE", "README.md", "pyproject.toml", "docs/library-guide.md", "docs/how-we-investigate-lfm3.html"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def publication_path(path):
    parts = PurePosixPath(path).parts
    require(bool(parts) and not path.startswith("/") and ".." not in parts and "\\" not in path, f"Invalid publication path: {path}")
    require(path in ROOT_FILES or path in WORKFLOWS or len(parts) > 1 and parts[0] in TOP_DIRS, f"Unexpected publication file: {path}")


def inspect(path, data):
    parts = PurePosixPath(path).parts
    lowered = {part.lower() for part in parts}
    require(not lowered & FORBIDDEN_PARTS, f"Forbidden path: {path}")
    require(PurePosixPath(path).suffix.lower() not in FORBIDDEN_SUFFIX, f"Forbidden artifact: {path}")
    require(not lowered & FORBIDDEN_NAMES, f"Machine-local receipt or archive material included: {path}")
    require(not any(part.startswith(".") for part in parts) or path in ROOT_FILES or path in WORKFLOWS, f"Unexpected hidden configuration: {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise AssertionError(f"Unexpected non-text content: {path}") from None
    for pattern, label in ((MACHINE, "Machine-specific reference"), (SECRET, "Possible credential"), (URL_CREDENTIAL, "Credential-bearing URL"), (STALE_URL, "Stale project URL")):
        for match in pattern.finditer(text):
            if pattern is MACHINE and (match.group("unix") or match.group("windows")) in SAMPLE_USERS:
                continue
            if pattern is URL_CREDENTIAL and (match.group("user"), match.group("password")) in SAMPLE_CREDENTIALS:
                continue
            line = text.count("\n", 0, match.start()) + 1
            require(False, f"{label}: {path}:{line}")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def audit_index(root):
    entries = git(root, "ls-files", "--stage", "-z").split(b"\0")
    tree = {}
    for entry in entries:
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, digest, stage = header.decode("ascii").split()
        name = raw_name.decode("utf-8")
        require(stage == "0", f"Unmerged index entry: {name}")
        require(mode in {"100644", "100755"}, f"Unexpected indexed file type: {name}")
        publication_path(name)
        data = git(root, "cat-file", "blob", digest)
        inspect(name, data)
        working = root / name
        require(not working.is_symlink() and working.is_file(), f"Missing or nonregular working file: {name}")
        require(data == working.read_bytes(), f"Staged/working mismatch: {name}")
        tree[name] = data
    require(tree, "Stage the proposed release files before auditing")
    return tree


def audit_remote(root, expected_url=None):
    require(expected_url is None or expected_url in AUTHORIZED_REMOTE_URLS, "Expected remote URL is not an authorized Glass-Box destination")
    remotes = git(root, "remote").decode().splitlines()
    require(len(remotes) <= 1, "Multiple remotes exist: review their authorization")
    require(not remotes or expected_url is not None, "A remote exists: supply its authorized --expected-remote-url")
    for name in remotes:
        for options in (("--all",), ("--push", "--all")):
            urls = git(root, "remote", "get-url", *options, name).decode().splitlines()
            require(urls == [expected_url], f"Unexpected remote endpoint: {name} ({'push' if '--push' in options else 'fetch'})")
    return len(remotes)


def audit_distributions(tree, wheel_path, sdist_path):
    config = tomllib.loads(tree["pyproject.toml"].decode("utf-8"))
    targets = config["tool"]["hatch"]["build"]["targets"]
    package = {name.removeprefix("src/"): data for name, data in tree.items() if name.startswith("src/lfm3lab/") and name.endswith(".py")}
    resources = {}
    for source, destination in targets["wheel"]["force-include"].items():
        require(source in tree, f"Unstaged wheel resource: {source}")
        require(destination not in package and destination not in resources, f"Duplicate wheel destination: {destination}")
        resources[destination] = tree[source]
    allowed_metadata = {"METADATA", "WHEEL", "entry_points.txt", "RECORD", "licenses/LICENSE", "licenses/NOTICE"}
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_names = wheel.namelist()
        require(len(wheel_names) == len(set(wheel_names)), "Duplicate wheel members")
        expected = set(package) | set(resources)
        actual = {name for name in wheel_names if ".dist-info/" not in name}
        require(actual == expected, f"Wheel package mismatch: {sorted(actual ^ expected)}")
        metadata_names = [name for name in wheel_names if name.endswith(".dist-info/METADATA")]
        require(len(metadata_names) == 1, "Expected exactly one wheel METADATA member")
        metadata_prefix = metadata_names[0].removesuffix("METADATA")
        metadata_members = {name.removeprefix(metadata_prefix) for name in wheel_names if ".dist-info/" in name}
        require(metadata_members == allowed_metadata, f"Wheel metadata mismatch: {sorted(metadata_members ^ allowed_metadata)}")
        for name in wheel_names:
            data = wheel.read(name)
            inspect(name, data)
            if name.startswith(metadata_prefix):
                tail = name.removeprefix(metadata_prefix)
                if tail.startswith("licenses/"):
                    require(data == tree[PurePosixPath(tail).name], f"License/source mismatch: {name}")
            elif name in package:
                require(data == package[name], f"Wheel/source mismatch: {name}")
            else:
                require(data == resources[name], f"Wheel/resource mismatch: {name}")
        metadata = email.message_from_bytes(wheel.read(metadata_names[0]))
        require(metadata["Name"] == "glass-box" and metadata["License-Expression"] == "Apache-2.0", "Unexpected wheel project or license metadata")
        require(set(metadata.get_all("License-File", [])) == {"LICENSE", "NOTICE"}, "Missing wheel license-file metadata")
        require(all("extra ==" in name for name in metadata.get_all("Requires-Dist", [])), "Unexpected mandatory dependencies")

    selection = targets["sdist"]["only-include"]
    exclusions = targets["sdist"].get("exclude", [])
    expected_sdist = {name for name in tree if any(name == prefix or name.startswith(prefix.rstrip("/") + "/") for prefix in selection) and not any(fnmatchcase(name, pattern) for pattern in exclusions)} | {"PKG-INFO"}
    # Hatch includes the repository ignore file as source-distribution metadata.
    expected_sdist |= {".gitignore"} & tree.keys()
    require(REQUIRED_SDIST <= expected_sdist, f"Required source distribution files excluded: {sorted(REQUIRED_SDIST - expected_sdist)}")
    with tarfile.open(sdist_path) as sdist:
        sdist_names = []
        prefix = None
        for member in sdist.getmembers():
            require(member.isfile(), f"Unexpected non-file sdist member: {member.name}")
            parts = PurePosixPath(member.name).parts
            require(len(parts) > 1 and not member.name.startswith("/") and ".." not in parts, f"Invalid sdist member: {member.name}")
            prefix = prefix or parts[0]
            require(parts[0] == prefix, f"Unexpected sdist root: {member.name}")
            name = "/".join(parts[1:])
            require(name in expected_sdist, f"Unexpected sdist file: {name}")
            if name != "PKG-INFO":
                publication_path(name)
            data = sdist.extractfile(member).read()
            inspect(name, data)
            if name != "PKG-INFO":
                require(data == tree[name], f"Sdist/source mismatch: {name}")
            sdist_names.append(name)
        require(len(sdist_names) == len(set(sdist_names)), "Duplicate sdist members")
        require(set(sdist_names) == expected_sdist, f"Sdist package mismatch: {sorted(set(sdist_names) ^ expected_sdist)}")
    return {"wheel_files": len(wheel_names), "sdist_files": len(sdist_names), "license": metadata["License-Expression"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--expected-remote-url", choices=sorted(AUTHORIZED_REMOTE_URLS), help="Exact owner-authorized fetch and push URL; omit before remote creation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tree = audit_index(root)
    remote_count = audit_remote(root, args.expected_remote_url)
    distributions = audit_distributions(tree, args.wheel, args.sdist)
    print(json.dumps({"passed": True, "staged_files": len(tree), **distributions,
                      "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                      "sdist_sha256": hashlib.sha256(args.sdist.read_bytes()).hexdigest(),
                      "machine_references_or_recognized_secrets": 0, "remote_count": remote_count,
                      "limits": "Pattern-based content audit; Git identity/history are untouched. Owner must confirm rights and authorize publication."}, indent=2))


if __name__ == "__main__":
    main()
