from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnPlan:
    name: str
    role: str                     # "numeric", "categorical", "text", "drop"
    enc_strategy: str | None = None   # "one_hot", "target", "hashing", ...
    impute_strategy: str | None = None
    scale_strategy: str | None = None # "standard", "robust", "none"
    text_strategy: str | None = None  # "tfidf", "embedding", "none"
    is_high_card: bool = False
    is_id_like: bool = False

@dataclass
class FeaturePlan:
    columns: list[ColumnPlan]
    target_name: str
    problem_type: str
