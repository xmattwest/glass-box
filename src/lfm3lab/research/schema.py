"""Versioned, JSON-only protocols for question-agnostic mechanistic studies."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields

SCHEMA_VERSION = 1
SPLITS = ("discovery", "validation", "holdout")
OBJECTIVES = ("token_margin", "token_logprob", "continuation_logprob")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("IDs must be 1..96 safe alphanumeric/underscore/dot/hyphen characters")


class JsonSpec:
    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            value = value.to_dict()
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown fields or invalid object for {cls.__name__}")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise ValueError(f"Invalid {cls.__name__}: {exc}") from exc
        return result.validate()

    def to_dict(self):
        return asdict(self)

    def validate(self):
        canonical_json(self.to_dict())
        return self


@dataclass
class Case(JsonSpec):
    id: str
    prompt: str
    target: int | str | None = None
    foil: int | str | None = None
    continuation: str | None = None
    corrupt_prompt: str | None = None
    split: str = "discovery"
    metadata: dict = field(default_factory=dict)

    def validate(self):
        super().validate()
        identifier(self.id)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Case.prompt must be nonempty text")
        if self.split not in SPLITS or not isinstance(self.metadata, dict):
            raise ValueError("Invalid case split or metadata")
        for name in ("target", "foil"):
            value = getattr(self, name)
            if value is not None and not (type(value) is int and value >= 0 or isinstance(value, str) and value):
                raise ValueError(f"{name} must be a nonnegative token ID or exact single-token text")
        for name in ("continuation", "corrupt_prompt"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be nonempty text when supplied")
        return self


@dataclass
class Condition(JsonSpec):
    id: str
    patch: dict = field(default_factory=dict)
    input_variant: str = "clean"
    description: str = ""

    def validate(self):
        super().validate()
        identifier(self.id)
        if self.input_variant not in ("clean", "corrupt") or not isinstance(self.description, str):
            raise ValueError("Invalid condition variant or description")
        # Keep protocol import model-free. Native-coordinate validation belongs
        # to the model adapter; preserve explicit unknown-capability errors there.
        if not isinstance(self.patch, dict) or set(self.patch) - {"rules", "norm_mode"}:
            raise ValueError("patch requires a PatchSpec JSON object")
        if self.patch.get("norm_mode", "native") not in ("native", "frozen"):
            raise ValueError("Invalid norm_mode")
        if not isinstance(self.patch.get("rules", []), list):
            raise ValueError("patch.rules must be a list")
        for rule in self.patch.get("rules", []):
            if not isinstance(rule, dict) or not isinstance(rule.get("unit"), str) or not isinstance(rule.get("op"), str):
                raise ValueError("Every patch rule requires unit/op strings")
            if "row" in rule:
                raise ValueError("Study conditions may not target internal batch rows")
        return self


@dataclass
class Objective(JsonSpec):
    kind: str = "token_margin"
    reduction: str = "sum"

    def validate(self):
        super().validate()
        if self.kind not in OBJECTIVES or self.reduction not in ("sum", "mean"):
            raise ValueError("Unsupported objective or reduction")
        if self.kind != "continuation_logprob" and self.reduction != "sum":
            raise ValueError("mean reduction only applies to continuation_logprob")
        return self


@dataclass
class Budget(JsonSpec):
    max_trials: int = 128
    max_seconds: float = 600
    max_input_tokens: int = 256

    def validate(self):
        super().validate()
        if type(self.max_trials) is not int or not 1 <= self.max_trials <= 100000:
            raise ValueError("max_trials must be 1..100000")
        if type(self.max_input_tokens) is not int or not 1 <= self.max_input_tokens <= 4096:
            raise ValueError("max_input_tokens must be 1..4096")
        if type(self.max_seconds) not in (float, int) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        return self


@dataclass
class Study(JsonSpec):
    id: str
    question: str
    prediction: str
    cases: list[Case]
    conditions: list[Condition]
    objective: Objective = field(default_factory=Objective)
    budget: Budget = field(default_factory=Budget)
    baseline_id: str = "clean"
    seed: int = 24
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.cases, list):
            self.cases = [Case.from_dict(c) for c in self.cases]
        if isinstance(self.conditions, list):
            self.conditions = [Condition.from_dict(c) for c in self.conditions]
        self.objective = Objective.from_dict(self.objective)
        self.budget = Budget.from_dict(self.budget)

    @property
    def fingerprint(self):
        return fingerprint({"schema_version": SCHEMA_VERSION, "study": self.to_dict()})

    def validate(self):
        super().validate()
        identifier(self.id)
        for name in ("question", "prediction"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Study requires a written {name} before execution")
        if not isinstance(self.metadata, dict) or type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid metadata or seed")
        for name, cls in (("cases", Case), ("conditions", Condition)):
            rows = getattr(self, name)
            if not isinstance(rows, list) or not rows or any(not isinstance(r, cls) for r in rows):
                raise ValueError(f"{name} must contain nonempty validated {cls.__name__} objects")
            for row in rows:
                row.validate()
            if len({r.id for r in rows}) != len(rows):
                raise ValueError(f"Duplicate {name} IDs")
        if self.baseline_id not in {r.id for r in self.conditions}:
            raise ValueError("baseline_id must name an explicit condition")
        secondary = self.metadata.get("secondary_comparisons", [])
        condition_ids = {r.id for r in self.conditions}
        if not isinstance(secondary, list) or any(
            not isinstance(pair, list) or len(pair) != 2 or
            any(not isinstance(name, str) or name not in condition_ids for name in pair) or
            pair[0] == pair[1] for pair in secondary
        ) or len({tuple(pair) for pair in secondary}) != len(secondary):
            raise ValueError("secondary_comparisons must be unique, distinct pairs of condition IDs")
        self.objective.validate()
        self.budget.validate()
        need_corrupt = any(c.input_variant == "corrupt" for c in self.conditions)
        for case in self.cases:
            if need_corrupt and case.corrupt_prompt is None:
                raise ValueError(f"Missing corrupt_prompt for case {case.id}")
            if self.objective.kind.startswith("token_") and case.target is None:
                raise ValueError("Token objectives require a fixed target for every case")
            if self.objective.kind == "token_margin" and (case.foil is None or case.target == case.foil):
                raise ValueError("Token margin requires distinct fixed target and foil")
            if self.objective.kind == "continuation_logprob" and case.continuation is None:
                raise ValueError("Continuation objective requires an explicit continuation")
        return self
