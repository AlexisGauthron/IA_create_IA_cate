from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ----------------------------
# Stats numériques
# ----------------------------
@dataclass
class NumericStats:
    """Statistiques de base pour une colonne numérique."""

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None  # médiane
    p75: float | None = None
    max: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None


# ----------------------------
# Stats catégorielles
# ----------------------------
@dataclass
class CategoricalStats:
    """Stats pour une variable catégorielle."""

    n_unique: int
    unique_ratio: float  # n_unique / n_rows
    top_values: list[dict[str, Any]] = field(default_factory=list)
    # ex: [{"value": "Support Technique", "count": 120, "freq": 0.12}, ...]

    n_rare_levels: int | None = None  # nb de modalités très rares
    rare_level_threshold: float | None = None  # ex: 0.01 => <1% = rare


# ----------------------------
# Stats texte
# ----------------------------
@dataclass
class TextStats:
    """Stats simples pour une colonne texte libre."""

    avg_char_length: float | None = None
    avg_token_length: float | None = None
    min_char_length: int | None = None
    max_char_length: int | None = None
