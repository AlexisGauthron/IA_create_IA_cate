# src/feature_engineering/hybrid/config.py
"""
Configuration pour le Feature Engineering Hybride (LLMFE + DFS).

Le mode hybride combine :
1. LLMFE : Features métier intelligentes (connaissance business)
2. DFS : Features structurelles (agrégations, interactions)

Priorité : Les features LLMFE sont prioritaires car elles apportent
la connaissance métier. DFS enrichit ensuite avec des agrégations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.feature_engineering.dfs.config import DFSConfig


@dataclass
class HybridConfig:
    """
    Configuration pour le Feature Engineering Hybride.

    Attributes:
        # Mode d'exécution
        enable_llmfe: Activer LLMFE (features métier)
        enable_dfs: Activer DFS (features structurelles)
        execution_order: Ordre d'exécution ('llmfe_first' ou 'parallel')

        # Configuration LLMFE
        llmfe_max_iterations: Nombre max d'itérations LLMFE
        llmfe_feature_format: Format des features ('basic', 'tags', 'hierarchical')
        llmfe_use_analysis: Utiliser le rapport d'analyse LLM

        # Configuration DFS
        dfs_config: Configuration DFS (ou nom d'une config prédéfinie)
        dfs_on_llmfe_features: Appliquer DFS sur les features LLMFE aussi

        # Sélection finale
        final_selection: Méthode de sélection finale
        max_features: Nombre max de features à garder
        feature_priority: Priorité en cas de conflit ('llmfe' ou 'dfs')

        # Évaluation
        eval_models: Modèles pour l'évaluation
        eval_metric: Métrique d'évaluation
        cv_folds: Nombre de folds pour la cross-validation
    """

    # Mode d'exécution
    enable_llmfe: bool = True
    enable_dfs: bool = True
    execution_order: Literal["llmfe_first", "parallel"] = "llmfe_first"

    # Configuration LLMFE
    llmfe_max_iterations: int = 10
    llmfe_feature_format: str = "basic"
    llmfe_use_analysis: bool = True
    llmfe_eval_models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])

    # Configuration DFS
    dfs_config: DFSConfig | str | None = None  # None = auto, str = nom prédéfini
    dfs_on_llmfe_features: bool = True  # Appliquer DFS sur features LLMFE

    # Sélection finale
    final_selection: Literal["importance", "hybrid", "none"] = "hybrid"
    max_features: int | None = 50
    feature_priority: Literal["llmfe", "dfs"] = "llmfe"  # LLMFE prioritaire
    correlation_threshold: float = 0.95  # Supprimer features trop corrélées

    # Évaluation
    eval_models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    eval_metric: str = "auto"
    cv_folds: int = 5

    # Autres
    verbose: bool = True
    random_state: int = 42

    def __post_init__(self):
        """Validation et configuration par défaut."""
        # Si dfs_config est un string, charger la config prédéfinie
        if isinstance(self.dfs_config, str):
            from src.feature_engineering.dfs.config import DFS_CONFIGS

            if self.dfs_config not in DFS_CONFIGS:
                raise ValueError(
                    f"Config DFS inconnue: {self.dfs_config}. "
                    f"Disponibles: {list(DFS_CONFIGS.keys())}"
                )
            self.dfs_config = DFS_CONFIGS[self.dfs_config]

        # Si dfs_config est None, utiliser synthetic_standard par défaut
        if self.dfs_config is None:
            self.dfs_config = DFSConfig(
                max_depth=2,
                agg_primitives=["mean", "std", "min", "max", "count"],
                trans_primitives=["is_null", "absolute"],
                feature_selection=False,  # Sélection gérée par HybridRunner
                create_synthetic_relations=True,
                max_synthetic_tables=5,
                verbose=False,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "enable_llmfe": self.enable_llmfe,
            "enable_dfs": self.enable_dfs,
            "execution_order": self.execution_order,
            "llmfe_max_iterations": self.llmfe_max_iterations,
            "llmfe_feature_format": self.llmfe_feature_format,
            "llmfe_use_analysis": self.llmfe_use_analysis,
            "dfs_config": self.dfs_config.to_dict() if self.dfs_config else None,
            "dfs_on_llmfe_features": self.dfs_on_llmfe_features,
            "final_selection": self.final_selection,
            "max_features": self.max_features,
            "feature_priority": self.feature_priority,
            "correlation_threshold": self.correlation_threshold,
            "eval_models": self.eval_models,
            "eval_metric": self.eval_metric,
            "cv_folds": self.cv_folds,
            "random_state": self.random_state,
        }


# Configurations prédéfinies
HYBRID_CONFIGS = {
    # Mode par défaut : équilibré
    "default": HybridConfig(
        enable_llmfe=True,
        enable_dfs=True,
        execution_order="llmfe_first",
        llmfe_max_iterations=10,
        dfs_config="synthetic_standard",
        max_features=50,
        feature_priority="llmfe",
    ),
    # Mode rapide : moins d'itérations
    "fast": HybridConfig(
        enable_llmfe=True,
        enable_dfs=True,
        execution_order="llmfe_first",
        llmfe_max_iterations=5,
        dfs_config="synthetic_basic",
        max_features=30,
        feature_priority="llmfe",
    ),
    # Mode exhaustif : plus de features
    "exhaustive": HybridConfig(
        enable_llmfe=True,
        enable_dfs=True,
        execution_order="llmfe_first",
        llmfe_max_iterations=15,
        dfs_config="synthetic_exhaustive",
        max_features=100,
        feature_priority="llmfe",
    ),
    # Mode LLMFE seul (fallback si pas de DFS)
    "llmfe_only": HybridConfig(
        enable_llmfe=True,
        enable_dfs=False,
        llmfe_max_iterations=15,
        max_features=None,
        feature_priority="llmfe",
    ),
    # Mode DFS seul (si pas de clé API LLM)
    "dfs_only": HybridConfig(
        enable_llmfe=False,
        enable_dfs=True,
        dfs_config="synthetic_exhaustive",
        max_features=75,
        feature_priority="dfs",
    ),
}


def get_hybrid_config(name: str) -> HybridConfig:
    """Récupère une configuration prédéfinie."""
    if name not in HYBRID_CONFIGS:
        raise ValueError(
            f"Config hybride inconnue: {name}. " f"Disponibles: {list(HYBRID_CONFIGS.keys())}"
        )
    return HYBRID_CONFIGS[name]
