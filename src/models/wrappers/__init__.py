"""
Wrappers pour les différents modèles ML.

Ce module expose tous les wrappers de modèles avec une interface uniforme.
"""

from src.models.wrappers.lightgbm_wrapper import LightGBMModel
from src.models.wrappers.sklearn_wrapper import (
    DecisionTreeModel,
    LogisticRegressionModel,
    RandomForestModel,
)
from src.models.wrappers.xgboost_wrapper import XGBoostModel

# CatBoost est optionnel (peut ne pas être installé)
try:
    from src.models.wrappers.catboost_wrapper import CatBoostModel

    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostModel = None  # type: ignore
    CATBOOST_AVAILABLE = False

__all__ = [
    "XGBoostModel",
    "LightGBMModel",
    "RandomForestModel",
    "DecisionTreeModel",
    "LogisticRegressionModel",
    "CatBoostModel",
    "CATBOOST_AVAILABLE",
]
