"""Model-free publication boundaries using real temporary Git and archives."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("glass_box_release_audit", ROOT / "scripts/release_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
PUBLIC_URL = "https://github.com/xmattwest/glass-box.git"


class ContentChecks(unittest.TestCase):
    def test_public_attribution_and_examples_do_not_bypass_credentials(self):
        examples = {
            "LICENSE": "Copyright The HuggingFace Team. https://www.apache.org/licenses/LICENSE-2.0",
            "NOTICE": "https://huggingface.co/LiquidAI/LFM3-1.7B/blob/revision/modeling_lfm3.py\nSeparate model license applies.",
            "src/lfm3lab/worker.py": "host = '127.0.0.1'\ntoken = None\n",
            "docs/guide.md": "http://localhost:18765 http://0.0.0.0:18765 http://127.0.0.1:18765\n/tmp/example ./my-studies " + "/home/" + "user/project\n" + "/Users/" + "<username>/project\n" + PUBLIC_URL,
            "docs/assets/workflow.svg": '<svg xmlns="http://www.w3.org/2000/svg"></svg>',
            ".github/workflows/check.yml": "name: CI\njobs: {}\n",
            ".github/workflows/pages.yml": "name: Pages\njobs: {}\n",
        }
        for name, text in examples.items():
            with self.subTest(name=name):
                audit.publication_path(name)
                line = text.count("\n") + 2
                with self.assertRaisesRegex(AssertionError, f"Possible credential: {name}:{line}"):
                    audit.inspect(name, (text + "\n" + "ghp_" + "A" * 24).encode())

    def test_machine_paths_fail_with_file_and_line_not_leaked_value(self):
        paths = [
            "/" + "Users/workstation-owner/project",
            "/" + "home/workstation-owner/project",
            "C:" + "\\Users\\workstation-owner\\project",
            "/private" + "/var/example",
            "/var" + "/folders/example",
            "~/" + "warm-precision/project",
        ]
        paths.append(json.dumps(paths[2]))
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaisesRegex(AssertionError, "Machine-specific reference: docs/guide.md:2") as caught:
                    audit.inspect("docs/guide.md", ("safe\n" + path).encode())
                self.assertNotIn("workstation-owner", str(caught.exception))

    def test_recognized_credentials_and_credential_urls_fail(self):
        credentials = [
            "hf_" + "A" * 24,
            "ghp_" + "A" * 24,
            "github_pat_" + "A" * 24,
            "AKIA" + "A" * 16,
            "ASIA" + "A" * 16,
            "sk-proj-" + "A" * 24,
            "-----BEGIN " + "ENCRYPTED PRIVATE KEY-----",
            'api_key = "' + "A" * 24 + '"',
            "https://" + "deploy:nonfixture-password@service.example/api",
        ]
        for credential in credentials:
            with self.subTest(kind=credential[:10]):
                with self.assertRaisesRegex(AssertionError, "credential|Credential"):
                    audit.inspect("docs/guide.md", credential.encode())

    def test_auth_fixtures_do_not_hide_actual_credentials(self):
        text = b'token="fixture-secret"\ntoken=None\nhttp://user:pass@localhost\n'
        with self.assertRaisesRegex(AssertionError, "Possible credential: src/lfm3lab/checks_worker.py:4"):
            audit.inspect("src/lfm3lab/checks_worker.py", text + b"ghp_" + b"A" * 24)

    def test_evidence_archive_and_hidden_configs_fail(self):
        paths = [
            "docs/verification-local.json", "docs/journal.jsonl", "docs/lab-log.md",
            "docs/evidence/report.json", "src/archive/helper.py", "examples/raw/trial.json",
            "docs/model.safetensors", "docs/notes.ipynb", "scripts/private.pem", "docs/.env",
        ]
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(AssertionError):
                    audit.inspect(path, b"{}")
        for path in (".github/workflows/other.yml", ".github/dependabot.yml", ".env", "../README.md"):
            with self.subTest(path=path):
                with self.assertRaises(AssertionError):
                    audit.publication_path(path)

    def test_stale_repository_urls_fail_but_public_usernames_pass(self):
        audit.inspect("docs/guide.md", b"xmattwest LiquidAI The HuggingFace Team")
        for repository in ("lfm3-research", "lfm3-glassbox"):
            with self.subTest(repository=repository):
                url = "https://github.com/" + "xmattwest/" + repository
                with self.assertRaisesRegex(AssertionError, "Stale project URL"):
                    audit.inspect("docs/guide.md", url.encode())

    def test_nontext_content_fails_with_member_name(self):
        with self.assertRaisesRegex(AssertionError, "Unexpected non-text content: docs/guide.md"):
            audit.inspect("docs/guide.md", b"\xff")


class GitChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="glass-box-audit-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git("init", "-q")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.PIPE)

    def stage(self, name, content):
        destination = self.root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        self.git("add", "--", name)

    def commit(self):
        self.git("-c", "user.name=Publication Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")

    def test_unchanged_committed_tree_is_audited(self):
        self.stage("README.md", b"Public source")
        self.stage("docs/guide.md", b"Portable docs")
        self.commit()
        self.assertEqual(audit.audit_index(self.root), {"README.md": b"Public source", "docs/guide.md": b"Portable docs"})

    def test_unchanged_committed_leak_is_not_hidden_by_safe_staged_change(self):
        self.stage("docs/guide.md", ("/" + "Users/workstation-owner/project").encode())
        self.commit()
        self.stage("README.md", b"Safe new publication text")
        with self.assertRaisesRegex(AssertionError, "Machine-specific reference: docs/guide.md:1"):
            audit.audit_index(self.root)

    def test_unchanged_committed_credential_fails(self):
        self.stage("docs/guide.md", ("ghp_" + "A" * 24).encode())
        self.commit()
        self.stage("README.md", b"Safe publication")
        with self.assertRaisesRegex(AssertionError, "Possible credential: docs/guide.md:1"):
            audit.audit_index(self.root)

    def test_empty_index_fails(self):
        with self.assertRaisesRegex(AssertionError, "Stage the proposed release files"):
            audit.audit_index(self.root)

    def test_unchanged_committed_evidence_fails(self):
        self.stage("docs/evidence/receipt.json", b"{}")
        self.commit()
        self.stage("README.md", b"Safe publication")
        with self.assertRaisesRegex(AssertionError, "Forbidden path"):
            audit.audit_index(self.root)

    def test_unstaged_modification_and_missing_working_file_fail(self):
        self.stage("README.md", b"Indexed source")
        (self.root / "README.md").write_bytes(b"Working source")
        with self.assertRaisesRegex(AssertionError, "Staged/working mismatch: README.md"):
            audit.audit_index(self.root)
        (self.root / "README.md").unlink()
        with self.assertRaisesRegex(AssertionError, "Missing or nonregular working file: README.md"):
            audit.audit_index(self.root)

    def test_indexed_symlinks_fail(self):
        self.stage("README.md", b"Public source")
        (self.root / "docs").mkdir()
        (self.root / "docs/guide.md").symlink_to("../README.md")
        self.git("add", "docs/guide.md")
        with self.assertRaisesRegex(AssertionError, "Unexpected indexed file type: docs/guide.md"):
            audit.audit_index(self.root)

    def test_reviewed_workflows_pass_and_other_workflows_fail(self):
        self.stage(".github/workflows/check.yml", b"name: CI\njobs: {}\n")
        self.stage(".github/workflows/pages.yml", b"name: Pages\njobs: {}\n")
        self.assertEqual(len(audit.audit_index(self.root)), 2)
        self.stage(".github/workflows/unchecked.yml", b"name: Unreviewed\n")
        with self.assertRaisesRegex(AssertionError, "Unexpected publication file"):
            audit.audit_index(self.root)

    def test_no_remote_and_explicit_authorized_remote_pass(self):
        self.assertEqual(audit.audit_remote(self.root), 0)
        self.git("remote", "add", "origin", PUBLIC_URL)
        with self.assertRaisesRegex(AssertionError, "expected-remote-url"):
            audit.audit_remote(self.root)
        self.assertEqual(audit.audit_remote(self.root, PUBLIC_URL), 1)

    def test_authorized_ssh_remote_passes_only_when_explicit(self):
        url = "git@github.com:xmattwest/glass-box.git"
        self.git("remote", "add", "origin", url)
        self.assertEqual(audit.audit_remote(self.root, url), 1)
        with self.assertRaisesRegex(AssertionError, "Unexpected remote endpoint"):
            audit.audit_remote(self.root, PUBLIC_URL)

    def test_wrong_remote_and_wrong_explicit_authorization_fail(self):
        self.git("remote", "add", "origin", "https://github.com/example/other.git")
        with self.assertRaisesRegex(AssertionError, "Unexpected remote endpoint"):
            audit.audit_remote(self.root, PUBLIC_URL)
        with self.assertRaisesRegex(AssertionError, "not an authorized Glass-Box destination"):
            audit.audit_remote(self.root, "https://github.com/example/other.git")

    def test_extra_remote_and_divergent_push_endpoint_fail(self):
        self.git("remote", "add", "origin", PUBLIC_URL)
        self.git("remote", "add", "other", PUBLIC_URL)
        with self.assertRaisesRegex(AssertionError, "Multiple remotes"):
            audit.audit_remote(self.root, PUBLIC_URL)
        self.git("remote", "remove", "other")
        self.git("remote", "set-url", "--push", "origin", "https://github.com/example/other.git")
        with self.assertRaisesRegex(AssertionError, r"Unexpected remote endpoint: origin \(push\)"):
            audit.audit_remote(self.root, PUBLIC_URL)


class DistributionChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="glass-box-package-audit-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.wheel = self.root / "glass_box-0.1.0-py3-none-any.whl"
        self.sdist = self.root / "glass_box-0.1.0.tar.gz"
        config = '''[tool.hatch.build.targets.wheel.force-include]
"docs/library-guide.md" = "lfm3lab/docs/library-guide.md"
[tool.hatch.build.targets.sdist]
only-include = ["README.md", "LICENSE", "NOTICE", "pyproject.toml", "docs", "src"]
exclude = []
'''
        self.tree = {
            "README.md": b"Public package", "LICENSE": b"Apache License, Version 2.0", "NOTICE": b"Retained decoder attribution; separate model terms.",
            "pyproject.toml": config.encode(), "src/lfm3lab/__init__.py": b'__version__ = "0.1.0"\n',
            "docs/library-guide.md": b"Portable agent guide", "docs/how-we-investigate-lfm3.html": b"<html></html>",
            "docs/additional.md": b"All selected docs must ship", ".github/workflows/check.yml": b"name: CI\n",
            ".gitignore": b".site/\ndist/\n",
        }
        metadata = b"Metadata-Version: 2.4\nName: glass-box\nVersion: 0.1.0\nLicense-Expression: Apache-2.0\nLicense-File: LICENSE\nLicense-File: NOTICE\nRequires-Dist: torch; extra == 'instruments'\n\n"
        prefix = "glass_box-0.1.0.dist-info/"
        self.wheel_members = {
            "lfm3lab/__init__.py": self.tree["src/lfm3lab/__init__.py"],
            "lfm3lab/docs/library-guide.md": self.tree["docs/library-guide.md"],
            prefix + "METADATA": metadata, prefix + "WHEEL": b"Wheel-Version: 1.0\n",
            prefix + "entry_points.txt": b"[console_scripts]\nglass-box = lfm3lab.research.cli:main\n",
            prefix + "RECORD": b"", prefix + "licenses/LICENSE": self.tree["LICENSE"], prefix + "licenses/NOTICE": self.tree["NOTICE"],
        }
        self.sdist_members = {name: data for name, data in self.tree.items() if not name.startswith(".github/")}
        self.sdist_members["PKG-INFO"] = metadata

    def write_archives(self, extra_sdist_member=None):
        with zipfile.ZipFile(self.wheel, "w") as wheel:
            for name, data in self.wheel_members.items():
                wheel.writestr(name, data)
        with tarfile.open(self.sdist, "w:gz") as sdist:
            for name, data in self.sdist_members.items():
                member = tarfile.TarInfo("glass_box-0.1.0/" + name)
                member.size = len(data)
                sdist.addfile(member, io.BytesIO(data))
            if extra_sdist_member is not None:
                sdist.addfile(extra_sdist_member)

    def run_audit(self):
        self.write_archives()
        return audit.audit_distributions(self.tree, self.wheel, self.sdist)

    def test_exact_members_source_resources_and_licenses_pass(self):
        result = self.run_audit()
        self.assertEqual(result["wheel_files"], len(self.wheel_members))
        self.assertEqual(result["sdist_files"], len(self.sdist_members))
        self.assertEqual(result["license"], "Apache-2.0")

    def test_wheel_source_resource_and_license_drift_fail(self):
        for name, error in (
            ("lfm3lab/__init__.py", "Wheel/source mismatch"),
            ("lfm3lab/docs/library-guide.md", "Wheel/resource mismatch"),
            ("glass_box-0.1.0.dist-info/licenses/NOTICE", "License/source mismatch"),
        ):
            with self.subTest(member=name):
                original = self.wheel_members[name]
                self.wheel_members[name] = b"Different bytes"
                with self.assertRaisesRegex(AssertionError, error):
                    self.run_audit()
                self.wheel_members[name] = original

    def test_extra_wheel_content_and_metadata_fail(self):
        for name, error in (("lfm3lab/untracked.py", "Wheel package mismatch"), ("other.dist-info/NOTICE", "Wheel metadata mismatch")):
            with self.subTest(member=name):
                self.wheel_members[name] = b"Unreviewed content"
                with self.assertRaisesRegex(AssertionError, error):
                    self.run_audit()
                del self.wheel_members[name]

    def test_leaked_wheel_metadata_and_mandatory_dependencies_fail(self):
        name = "glass_box-0.1.0.dist-info/METADATA"
        original = self.wheel_members[name]
        self.wheel_members[name] = original + ("/" + "Users/workstation-owner/build").encode()
        with self.assertRaisesRegex(AssertionError, "Machine-specific reference"):
            self.run_audit()
        self.wheel_members[name] = original.replace(b"extra == 'instruments'", b"python_version >= '3.12'")
        with self.assertRaisesRegex(AssertionError, "Unexpected mandatory dependencies"):
            self.run_audit()

    def test_missing_nonrequired_sdist_member_fails(self):
        del self.sdist_members["docs/additional.md"]
        with self.assertRaisesRegex(AssertionError, "Sdist package mismatch.*docs/additional.md"):
            self.run_audit()

    def test_untracked_sdist_member_and_source_drift_fail(self):
        self.sdist_members["docs/untracked.md"] = b"Unstaged doc"
        with self.assertRaisesRegex(AssertionError, "Unexpected sdist file"):
            self.run_audit()
        del self.sdist_members["docs/untracked.md"]
        self.sdist_members["docs/additional.md"] = b"Different bytes"
        with self.assertRaisesRegex(AssertionError, "Sdist/source mismatch"):
            self.run_audit()

    def test_sdist_nonfile_and_duplicate_members_fail(self):
        link = tarfile.TarInfo("glass_box-0.1.0/docs/link.md")
        link.type = tarfile.SYMTYPE
        link.linkname = "../README.md"
        self.write_archives(link)
        with self.assertRaisesRegex(AssertionError, "Unexpected non-file sdist member"):
            audit.audit_distributions(self.tree, self.wheel, self.sdist)
        duplicate = tarfile.TarInfo("glass_box-0.1.0/PKG-INFO")
        self.write_archives(duplicate)
        with self.assertRaisesRegex(AssertionError, "Duplicate sdist members"):
            audit.audit_distributions(self.tree, self.wheel, self.sdist)

    def test_sdist_generated_metadata_is_content_audited(self):
        self.sdist_members["PKG-INFO"] += ("hf_" + "A" * 24).encode()
        with self.assertRaisesRegex(AssertionError, "Possible credential: PKG-INFO"):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
