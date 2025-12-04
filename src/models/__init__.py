"""
Module centralisé pour les modèles ML.

Ce module fournit une interface unifiée pour tous les modèles ML utilisés
dans le pipeline AutoML. Il est conçu pour être réutilisé par :
- LLMFE (évaluation multi-modèle des features)
- CASH (sélection d'algorithme + HPO)
- Landmarking (méta-learning)
- Screening (évaluation rapide)

Architecture:
    src/models/
    ├── base.py              # Interface BaseModel
    ├── registry.py          # Registre centralisé get_model()
    ├── config.py            # Configurations et presets
    ├── wrappers/            # Wrappers par modèle
    │   ├── xgboost_wrapper.py
    │   ├── lightgbm_wrapper.py
    │   ├── sklearn_wrapper.py
    │   └── catboost_wrapper.py
    └── evaluation/          # Évaluation
        ├── metrics.py       # Métriques standardisées
        └── cross_validator.py

Example:
    >>> from src.models import get_model, get_all_models, CrossValidator
    >>>
    >>> # Récupérer un modèle
    >>> model = get_model("xgboost", is_regression=False)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>>
    >>> # Évaluer en cross-validation
    >>> cv = CrossValidator(n_folds=5)
    >>> result = cv.evaluate(model, X, y, metric="f1")
    >>> print(f"Score: {result.mean:.4f}")
    >>>
    >>> # Évaluation multi-modèle (pour LLMFE)
    >>> models = get_all_models(is_regression=False)
    >>> multi_result = cv.evaluate_multi_model(models, X, y)
    >>> print(f"Meilleur modèle: {multi_result.best_model}")
"""

# Base
from src.models.base import BaseModel

# Registry
from src.models.registry import (
    ALL_TREE_MODELS,
    CATBOOST_AVAILABLE,
    DEFAULT_MODELS,
    FAST_MODELS,
    MODEL_REGISTRY,
    get_all_models,
    get_model,
    get_models,
    is_model_available,
    list_models,
    register_model,
)

# Wrappers (pour accès direct si nécessaire)
from src.models.wrappers import (
    DecisionTreeModel,
    LightGBMModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
)

# CatBoost optionnel
try:
    from src.models.wrappers import CatBoostModel
except ImportError:
    CatBoostModel = None  # type: ignore

# Evaluation
# Config
from src.models.config import (
    DEFAULT_CONFIG,
    HPO_CONFIG,
    LANDMARKING_CONFIG,
    LLMFE_CONFIG,
    SCREENING_CONFIG,
    DefaultConfig,
    HPOConfig,
    LandmarkingConfig,
    LLMFEConfig,
    ModelConfig,
    ScreeningConfig,
    get_config,
    list_presets,
)
from src.models.evaluation import (
    CrossValidator,
    CVResult,
    MultiModelCVResult,
    compute_all_metrics,
    get_default_metric,
    get_metric,
    get_scorer,
    list_metrics,
)

__all__ = [
    # Base
    "BaseModel",
    # Registry
    "MODEL_REGISTRY",
    "CATBOOST_AVAILABLE",
    "FAST_MODELS",
    "DEFAULT_MODELS",
    "ALL_TREE_MODELS",
    "get_model",
    "get_models",
    "get_all_models",
    "list_models",
    "register_model",
    "is_model_available",
    # Wrappers
    "XGBoostModel",
    "LightGBMModel",
    "RandomForestModel",
    "DecisionTreeModel",
    "LogisticRegressionModel",
    "CatBoostModel",
    # Evaluation
    "CrossValidator",
    "CVResult",
    "MultiModelCVResult",
    "get_metric",
    "get_scorer",
    "get_default_metric",
    "list_metrics",
    "compute_all_metrics",
    # Config
    "ModelConfig",
    "ScreeningConfig",
    "DefaultConfig",
    "HPOConfig",
    "LLMFEConfig",
    "LandmarkingConfig",
    "SCREENING_CONFIG",
    "DEFAULT_CONFIG",
    "HPO_CONFIG",
    "LLMFE_CONFIG",
    "LANDMARKING_CONFIG",
    "get_config",
    "list_presets",
]
