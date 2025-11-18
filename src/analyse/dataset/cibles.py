from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any


@dataclass
class TargetSummaryForLLM:
    """Description détaillée de la cible pour le LLM."""
    name: str

    problem_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "multilabel_classification",
        "regression",
        "unknown"
    ] = "unknown"

    pandas_dtype: str = "object"
    inferred_target_type: Literal["numeric", "categorical", "text", "mixed"] = "categorical"

    n_rows: int = 0
    n_unique: int = 0
    missing_rate: float = 0.0

    # Distribution des classes (si catégorielle)
    class_counts: Optional[Dict[str, int]] = None
    class_proportions: Optional[Dict[str, float]] = None
    most_frequent_classes: Optional[List[Dict[str, Any]]] = None
    # ex: [{"value": "Support Technique", "count": 800, "freq": 0.40}, ...]

    # Indicateurs d’imbalance
    imbalance_ratio: Optional[float] = None  # maj_class / min_class
    is_imbalanced: Optional[bool] = None

    # Pour binaire
    positive_class: Optional[str] = None

    # Notes / avertissements
    notes: List[str] = field(default_factory=list)
