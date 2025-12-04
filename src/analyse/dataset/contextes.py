from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetContextForLLM:
    """Métadonnées métier et techniques générales sur le dataset."""

    name: str
    business_description: str | None = None
    # ex: "Tickets de support client avec classification en 18 catégories métiers."

    metric: str | None = None
    # ex: "f1_macro", "accuracy", "roc_auc", "rmse"

    is_time_dependent: bool = False
    time_column: str | None = None
    # ex: "date_ticket"

    primary_keys: list[str] = field(default_factory=list)
    group_keys: list[str] = field(default_factory=list)
    # ex: primary_keys=["id_ticket"], group_keys=["id_client"]
