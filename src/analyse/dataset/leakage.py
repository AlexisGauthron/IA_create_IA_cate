# src/analyse/dataset/leakage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class LeakageSignalForLLM:
    feature: str
    target: str
    correlation: float
    note: str
    severity: Literal["info", "warning", "strong"] = "strong"
