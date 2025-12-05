# src/feature_engineering/dfs/config.py
"""
Configuration pour Deep Feature Synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DFSConfig:
    """
    Configuration pour le Deep Feature Synthesis.

    Attributes:
        max_depth: Profondeur maximale des features (1-3 recommandé)
        agg_primitives: Primitives d'agrégation à utiliser
        trans_primitives: Primitives de transformation à utiliser
        max_features: Nombre maximum de features à générer (None = illimité)
        feature_selection: Activer la sélection automatique des features
        selection_method: Méthode de sélection ('importance', 'correlation', 'rfe')
        selection_threshold: Seuil pour la sélection (importance min ou correlation max)
        n_jobs: Nombre de jobs parallèles (-1 = tous les CPU)
        chunk_size: Taille des chunks pour le traitement par lots
        verbose: Afficher les logs
    """

    # Paramètres DFS
    max_depth: int = 2
    agg_primitives: list[str] | None = None
    trans_primitives: list[str] | None = None
    max_features: int | None = None

    # Sélection des features
    feature_selection: bool = True
    selection_method: str = "importance"  # 'importance', 'correlation', 'rfe', 'hybrid'
    selection_threshold: float = 0.01  # Importance minimale
    correlation_threshold: float = 0.95  # Corrélation maximale entre features
    top_k_features: int | None = None  # Garder les top K features (optionnel)

    # Performance
    n_jobs: int = -1
    chunk_size: int | None = None

    # Relations synthétiques (pour single-table)
    create_synthetic_relations: bool = False  # Activer les relations synthétiques
    synthetic_groupby_cols: list[str] | None = None  # Colonnes pour grouper (auto si None)
    min_group_size: int = 5  # Taille minimale d'un groupe pour créer une relation
    max_synthetic_tables: int = 5  # Nombre max de tables synthétiques

    # Évaluation
    eval_models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    eval_metric: str = "auto"
    cv_folds: int = 5

    # Autres
    verbose: bool = True
    random_state: int = 42

    def __post_init__(self):
        """Validation et valeurs par défaut."""
        if self.max_depth < 1 or self.max_depth > 4:
            raise ValueError("max_depth doit être entre 1 et 4")

        if self.selection_method not in ["importance", "correlation", "rfe", "hybrid"]:
            raise ValueError(
                f"selection_method invalide: {self.selection_method}. "
                "Choix: importance, correlation, rfe, hybrid"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "max_depth": self.max_depth,
            "agg_primitives": self.agg_primitives,
            "trans_primitives": self.trans_primitives,
            "max_features": self.max_features,
            "feature_selection": self.feature_selection,
            "selection_method": self.selection_method,
            "selection_threshold": self.selection_threshold,
            "correlation_threshold": self.correlation_threshold,
            "top_k_features": self.top_k_features,
            "create_synthetic_relations": self.create_synthetic_relations,
            "synthetic_groupby_cols": self.synthetic_groupby_cols,
            "min_group_size": self.min_group_size,
            "max_synthetic_tables": self.max_synthetic_tables,
            "eval_models": self.eval_models,
            "eval_metric": self.eval_metric,
            "cv_folds": self.cv_folds,
            "random_state": self.random_state,
        }


# Configurations prédéfinies
DFS_CONFIGS = {
    "minimal": DFSConfig(
        max_depth=1,
        agg_primitives=["mean", "sum", "count"],
        trans_primitives=["year", "month"],
        feature_selection=True,
        selection_method="importance",
    ),
    "standard": DFSConfig(
        max_depth=2,
        agg_primitives=None,  # Utilise les defaults
        trans_primitives=None,
        feature_selection=True,
        selection_method="importance",
    ),
    "exhaustive": DFSConfig(
        max_depth=3,
        agg_primitives=None,
        trans_primitives=None,
        feature_selection=True,
        selection_method="hybrid",
        top_k_features=100,
    ),
    # Configurations avec relations synthétiques (pour single-table)
    "synthetic_basic": DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "std", "min", "max", "count"],
        trans_primitives=["is_null", "absolute"],
        feature_selection=True,
        selection_method="hybrid",
        top_k_features=30,
        create_synthetic_relations=True,
        max_synthetic_tables=3,
    ),
    "synthetic_standard": DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "std", "min", "max", "count", "num_unique", "median"],
        trans_primitives=["is_null", "absolute", "percentile"],
        feature_selection=True,
        selection_method="hybrid",
        top_k_features=50,
        create_synthetic_relations=True,
        max_synthetic_tables=5,
    ),
    "synthetic_exhaustive": DFSConfig(
        max_depth=3,
        agg_primitives=None,  # Tous les defaults
        trans_primitives=None,
        feature_selection=True,
        selection_method="hybrid",
        top_k_features=100,
        create_synthetic_relations=True,
        max_synthetic_tables=8,
    ),
}


def get_dfs_config(name: str) -> DFSConfig:
    """Récupère une configuration prédéfinie."""
    if name not in DFS_CONFIGS:
        raise ValueError(f"Config inconnue: {name}. Disponibles: {list(DFS_CONFIGS.keys())}")
    return DFS_CONFIGS[name]
