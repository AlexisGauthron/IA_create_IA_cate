from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TargetSummaryForLLM:
    """Description détaillée de la cible pour le LLM."""

    name: str

    problem_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "multilabel_classification",
        "regression",
        "unknown",
    ] = "unknown"

    pandas_dtype: str = "object"
    inferred_target_type: Literal["numeric", "categorical", "text", "mixed"] = "categorical"

    n_rows: int = 0
    n_unique: int = 0
    missing_rate: float = 0.0

    # Distribution des classes (si catégorielle)
    class_counts: dict[str, int] | None = None
    class_proportions: dict[str, float] | None = None
    most_frequent_classes: list[dict[str, Any]] | None = None
    # ex: [{"value": "Support Technique", "count": 800, "freq": 0.40}, ...]

    # Indicateurs d’imbalance
    imbalance_ratio: float | None = None  # maj_class / min_class
    is_imbalanced: bool | None = None

    # Pour binaire
    positive_class: str | None = None

    # Notes / avertissements
    notes: list[str] = field(default_factory=list)
