# src/feature_engineering/dfs/__init__.py
"""
Module Deep Feature Synthesis (DFS) pour le feature engineering automatique.

Basé sur Featuretools, ce module implémente l'algorithme DFS développé au MIT
pour générer automatiquement des features à partir de données tabulaires.

Références:
- https://featuretools.alteryx.com/en/stable/getting_started/afe.html
- https://www.kdnuggets.com/2018/02/deep-feature-synthesis-automated-feature-engineering.html
"""

from __future__ import annotations

from src.feature_engineering.dfs.config import DFSConfig
from src.feature_engineering.dfs.primitives import (
    AGGREGATION_PRIMITIVES,
    TRANSFORM_PRIMITIVES,
    get_primitives_for_task,
)
from src.feature_engineering.dfs.runner import DFSRunner, run_dfs
from src.feature_engineering.dfs.selection import FeatureSelector

__all__ = [
    "DFSConfig",
    "DFSRunner",
    "run_dfs",
    "FeatureSelector",
    "AGGREGATION_PRIMITIVES",
    "TRANSFORM_PRIMITIVES",
    "get_primitives_for_task",
]
