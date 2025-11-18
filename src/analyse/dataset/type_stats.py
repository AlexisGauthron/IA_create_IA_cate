from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any


# ----------------------------
# Stats numériques
# ----------------------------
@dataclass
class NumericStats:
    """Statistiques de base pour une colonne numérique."""
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None  # médiane
    p75: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

# ----------------------------
# Stats catégorielles
# ----------------------------
@dataclass
class CategoricalStats:
    """Stats pour une variable catégorielle."""
    n_unique: int
    unique_ratio: float           # n_unique / n_rows
    top_values: List[Dict[str, Any]] = field(default_factory=list)
    # ex: [{"value": "Support Technique", "count": 120, "freq": 0.12}, ...]

    n_rare_levels: Optional[int] = None    # nb de modalités très rares
    rare_level_threshold: Optional[float] = None  # ex: 0.01 => <1% = rare


# ----------------------------
# Stats texte
# ----------------------------
@dataclass
class TextStats:
    """Stats simples pour une colonne texte libre."""
    avg_char_length: Optional[float] = None
    avg_token_length: Optional[float] = None
    min_char_length: Optional[int] = None
    max_char_length: Optional[int] = None

