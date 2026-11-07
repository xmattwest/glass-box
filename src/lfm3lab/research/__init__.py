"""Reproducible mechanistic research. No model loading at import time."""
from .schema import Budget, Case, Condition, Objective, Study

__all__ = ["Budget", "Case", "Condition", "Objective", "Study", "ResearchSession", "run_study", "build_report"]


def __getattr__(name):
    if name == "ResearchSession":
        from .session import ResearchSession
        return ResearchSession
    if name == "run_study":
        from .runner import run_study
        return run_study
    if name == "build_report":
        from .report import build_report
        return build_report
    raise AttributeError(name)
