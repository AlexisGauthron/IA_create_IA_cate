from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any


@dataclass
class DatasetContextForLLM:
    """Métadonnées métier et techniques générales sur le dataset."""
    name: str
    business_description: Optional[str] = None
    # ex: "Tickets de support client avec classification en 18 catégories métiers."

    metric: Optional[str] = None
    # ex: "f1_macro", "accuracy", "roc_auc", "rmse"

    is_time_dependent: bool = False
    time_column: Optional[str] = None
    # ex: "date_ticket"

    primary_keys: List[str] = field(default_factory=list)
    group_keys: List[str] = field(default_factory=list)
    # ex: primary_keys=["id_ticket"], group_keys=["id_client"]
