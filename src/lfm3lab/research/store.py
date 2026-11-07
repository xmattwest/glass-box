"""Model-free, locked and hash-verified evidence storage (format version 1)."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from .schema import SCHEMA_VERSION, Study, canonical_json, fingerprint


class EvidenceError(ValueError):
    """An incompatible or damaged evidence directory must not be reused."""


class OutputLocked(RuntimeError):
    pass


def backend_identity(status):
    """Keep scientific identity, not loaded state, uptime, URL or transport state."""
    if not isinstance(status, dict):
        raise EvidenceError("Backend status must be a JSON object")
    source = status.get("identity", status)
    if not isinstance(source, dict):
        raise EvidenceError("Backend identity must be a JSON object")
    keys = ("model", "model_id", "repo_id", "checkpoint", "backend", "revision", "device", "dtype",
            "adapter", "adapter_version", "implementation", "implementation_version")
    identity = {key: source[key] for key in keys if key in source}
    if not all(key in identity for key in ("revision", "device", "dtype", "adapter", "implementation")):
        raise EvidenceError("Backend status needs revision, device, dtype, adapter and implementation identity")
    canonical_json(identity)
    return identity


@contextlib.contextmanager
def output_lock(output):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    # Never unlink this inode: unlinking creates races with existing waiters.
    with (path / ".lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OutputLocked(f"Evidence output is locked: {path}") from exc
        try:
            yield path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new_json(path, value):
    """Exclusive immutable receipt creation; fsync before it is journaled."""
    data = (canonical_json(value) + "\n").encode()
    with Path(path).open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_dir(Path(path).parent)
    return hashlib.sha256(data).hexdigest()


def write_report_json(path, value):
    """Only derived reports may be replaced, never protocol or raw evidence."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_dir(path.parent)


def trial_key(split, case_id, condition_id):
    return canonical_json([split, case_id, condition_id])


class EvidenceStore:
    """Caller must hold output_lock for every read or mutation."""

    def __init__(self, output):
        self.path = Path(output)
        self.events = []
        self.trials = {}
        self.responses = {}

    def initialize(self, study, identity, *, resume):
        protocol_path = self.path / "protocol.json"
        if protocol_path.exists():
            if not resume:
                raise EvidenceError("Output already contains a protocol; use resume or a new directory")
            self.load()
            if self.protocol["study_fingerprint"] != study.fingerprint:
                raise EvidenceError("Study differs from immutable protocol")
            if self.protocol["backend_identity"] != identity:
                raise EvidenceError("Backend identity differs from immutable protocol")
            return
        if any(p.name != ".lock" for p in self.path.iterdir()):
            raise EvidenceError("Refusing nonempty output without an intact protocol")
        self.protocol = {"format_version": 1, "schema_version": SCHEMA_VERSION,
                         "study": study.to_dict(), "study_fingerprint": study.fingerprint,
                         "backend_identity": identity, "created_at": time.time()}
        self.protocol["snapshot_hash"] = fingerprint(self.protocol)
        write_new_json(protocol_path, self.protocol)
        (self.path / "raw").mkdir()
        _sync_dir(self.path)
        self.load()

    def load(self):
        try:
            self._load()
        except (OSError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, EvidenceError):
                raise
            raise EvidenceError(f"Invalid evidence: {exc}") from exc
        return self

    def _load(self):
        self.protocol = json.loads((self.path / "protocol.json").read_text())
        core = {k: v for k, v in self.protocol.items() if k != "snapshot_hash"}
        if fingerprint(core) != self.protocol["snapshot_hash"]:
            raise EvidenceError("Protocol snapshot hash mismatch")
        if self.protocol["format_version"] != 1 or self.protocol["schema_version"] != SCHEMA_VERSION:
            raise EvidenceError("Unsupported evidence version")
        self.study = Study.from_dict(self.protocol["study"])
        if self.study.fingerprint != self.protocol["study_fingerprint"]:
            raise EvidenceError("Protocol study fingerprint mismatch")
        self.events, self.trials, self.responses = [], {}, {}
        journal = self.path / "journal.jsonl"
        if journal.exists():
            data = journal.read_bytes()
            if data and not data.endswith(b"\n"):
                raise EvidenceError("Truncated journal; preserve output and investigate, do not silently repair")
            for line in data.splitlines():
                event = json.loads(line)
                self._accept(event)
        # Orphan artifacts from a crash before completion are preserved, not
        # accepted as results. Their intents remain interrupted attempts.

    def _accept(self, event):
        body = {k: v for k, v in event.items() if k != "event_hash"}
        previous = self.events[-1]["event_hash"] if self.events else self.protocol["snapshot_hash"]
        if event["seq"] != len(self.events) or event["previous_hash"] != previous or fingerprint(body) != event["event_hash"]:
            raise EvidenceError("Journal hash chain mismatch")
        key = event["key"]
        if event["kind"] == "intent":
            if key in self.trials:
                raise EvidenceError("Duplicate trial intent")
            valid = {trial_key(c.split, c.id, d.id) for c in self.study.cases for d in self.study.conditions}
            if key not in valid:
                raise EvidenceError("Trial outside frozen protocol")
            if type(event["reserved_seconds"]) not in (int, float) or event["reserved_seconds"] <= 0:
                raise EvidenceError("Invalid time reservation")
            canonical_json(event)
            self.trials[key] = {"intent": event, "terminal": None}
        elif event["kind"] in ("result", "failure"):
            trial = self.trials.get(key)
            if trial is None or trial["terminal"] is not None:
                raise EvidenceError("Terminal event without unique pending intent")
            if type(event["elapsed_s"]) not in (int, float) or event["elapsed_s"] < 0:
                raise EvidenceError("Invalid elapsed budget charge")
            if event["kind"] == "result" or "artifact" in event:
                expected = f"raw/{trial['intent']['seq']:08d}.json"
                if event["artifact"] != expected:
                    raise EvidenceError("Unsafe or unexpected raw artifact path")
                data = (self.path / expected).read_bytes()
                if hashlib.sha256(data).hexdigest() != event["artifact_hash"]:
                    raise EvidenceError("Raw artifact hash mismatch")
                receipt = json.loads(data)
                if receipt["key"] != key or receipt["payload"] != trial["intent"]["payload"]:
                    raise EvidenceError("Receipt does not match intent")
                if event["kind"] == "result":
                    self.responses[key] = receipt["response"]
            trial["terminal"] = event
        else:
            raise EvidenceError("Unknown journal event")
        canonical_json(event)  # Reject nonfinite JSON, including durations.
        self.events.append(event)

    def append(self, kind, key, **details):
        event = {"seq": len(self.events), "previous_hash": self.events[-1]["event_hash"] if self.events else self.protocol["snapshot_hash"],
                 "kind": kind, "key": key, "at": time.time(), **details}
        event["event_hash"] = fingerprint(event)
        # Validate first, then durably append. Failure cannot silently consume a
        # partial journal line as if it were completed work.
        self._accept(event)
        with (self.path / "journal.jsonl").open("ab") as handle:
            handle.write((canonical_json(event) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        _sync_dir(self.path)
        return event

    def recover_interrupted(self):
        for key, trial in list(self.trials.items()):
            if trial["terminal"] is None:
                self.append("failure", key, error_type="InterruptedAttempt",
                            error="No durable completion. Not retried; remaining time reservation charged conservatively.",
                            elapsed_s=trial["intent"]["reserved_seconds"], interrupted=True)

    def budget_used(self):
        return {"attempts": len(self.trials), "elapsed_s": sum(
            t["terminal"]["elapsed_s"] if t["terminal"] else t["intent"]["reserved_seconds"]
            for t in self.trials.values())}
