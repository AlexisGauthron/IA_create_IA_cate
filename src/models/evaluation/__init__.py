"""
Module d'évaluation unifié pour les modèles ML.

Ce module fournit :
- Métriques standardisées (classification et régression)
- CrossValidator unifié pour évaluation cohérente
- Scorer combiné pour évaluation multi-métrique
"""

from src.models.evaluation.cross_validator import (
    CrossValidator,
    CVResult,
    MultiModelCVResult,
)
from src.models.evaluation.metrics import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    compute_all_metrics,
    get_default_metric,
    get_metric,
    get_scorer,
    is_proba_metric,
    list_metrics,
)

__all__ = [
    # Métriques
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "get_metric",
    "get_scorer",
    "get_default_metric",
    "list_metrics",
    "is_proba_metric",
    "compute_all_metrics",
    # CrossValidator
    "CrossValidator",
    "CVResult",
    "MultiModelCVResult",
]
