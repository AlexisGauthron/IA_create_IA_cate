from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any


@dataclass
class BasicDatasetStats:
    """Stats globales du dataset, utiles pour donner du contexte au LLM."""
    n_rows: int
    n_columns: int
    n_features: int
    n_numeric_features: int
    n_categorical_features: int
    n_text_features: int
    n_datetime_features: int

    total_missing_cells: int
    missing_cell_ratio: float    # sur l'ensemble du tableau

    n_duplicate_rows: int
    duplicate_row_ratio: float

    memory_mb: Optional[float] = None  # taille approx. en mémoire
