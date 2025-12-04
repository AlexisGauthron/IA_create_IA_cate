"""
Registre centralisé des modèles ML.

Ce module fournit un point d'accès unique pour récupérer n'importe quel modèle
par son nom. Il permet de switcher facilement entre différents algorithmes.

Example:
    >>> from src.models.registry import get_model, list_models
    >>> print(list_models())  # ['xgboost', 'lightgbm', 'randomforest', ...]
    >>> model = get_model("xgboost", is_regression=False)
    >>> model.fit(X_train, y_train)
"""

from typing import Type

from src.models.base import BaseModel
from src.models.wrappers.lightgbm_wrapper import LightGBMModel
from src.models.wrappers.sklearn_wrapper import (
    DecisionTreeModel,
    LogisticRegressionModel,
    RandomForestModel,
)
from src.models.wrappers.xgboost_wrapper import XGBoostModel

# Registre principal des modèles
MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "randomforest": RandomForestModel,
    "decisiontree": DecisionTreeModel,
    "logistic": LogisticRegressionModel,
}

# Ajouter CatBoost si disponible
try:
    from src.models.wrappers.catboost_wrapper import CatBoostModel

    MODEL_REGISTRY["catboost"] = CatBoostModel
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def get_model(name: str, is_regression: bool = False, **kwargs) -> BaseModel:
    """
    Récupère un modèle par son nom.

    Args:
        name: Nom du modèle ('xgboost', 'lightgbm', 'randomforest', etc.)
        is_regression: True pour régression, False pour classification
        **kwargs: Hyperparamètres additionnels à passer au modèle

    Returns:
        Instance du modèle (non entraîné)

    Raises:
        ValueError: Si le nom du modèle est inconnu

    Example:
        >>> model = get_model("xgboost", is_regression=False)
        >>> model = get_model("lightgbm", is_regression=True, n_estimators=200)
    """
    name_lower = name.lower()

    if name_lower not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Modèle inconnu: '{name}'. " f"Modèles disponibles: {available}")

    return MODEL_REGISTRY[name_lower](is_regression=is_regression, **kwargs)


def get_models(names: list[str], is_regression: bool = False, **kwargs) -> list[BaseModel]:
    """
    Récupère plusieurs modèles par leurs noms.

    Args:
        names: Liste des noms de modèles
        is_regression: True pour régression, False pour classification
        **kwargs: Hyperparamètres communs à tous les modèles

    Returns:
        Liste d'instances de modèles (non entraînés)

    Example:
        >>> models = get_models(["xgboost", "lightgbm", "randomforest"])
    """
    return [get_model(name, is_regression, **kwargs) for name in names]


def get_all_models(is_regression: bool = False, **kwargs) -> list[BaseModel]:
    """
    Récupère tous les modèles disponibles.

    Utile pour le screening rapide ou le benchmarking.

    Args:
        is_regression: True pour régression, False pour classification
        **kwargs: Hyperparamètres communs à tous les modèles

    Returns:
        Liste de tous les modèles disponibles (non entraînés)

    Example:
        >>> all_models = get_all_models(is_regression=False)
        >>> for model in all_models:
        ...     print(model.get_name())
    """
    return [
        model_class(is_regression=is_regression, **kwargs)
        for model_class in MODEL_REGISTRY.values()
    ]


def list_models() -> list[str]:
    """
    Liste les noms de tous les modèles disponibles.

    Returns:
        Liste des noms de modèles

    Example:
        >>> print(list_models())
        ['xgboost', 'lightgbm', 'randomforest', 'decisiontree', 'logistic', 'catboost']
    """
    return list(MODEL_REGISTRY.keys())


def register_model(name: str, model_class: Type[BaseModel]) -> None:
    """
    Enregistre un nouveau modèle dans le registre.

    Permet d'étendre le registre avec des modèles custom.

    Args:
        name: Nom unique pour le modèle
        model_class: Classe du modèle (doit hériter de BaseModel)

    Raises:
        TypeError: Si model_class n'hérite pas de BaseModel

    Example:
        >>> class MyCustomModel(BaseModel):
        ...     # Implementation
        ...     pass
        >>> register_model("mymodel", MyCustomModel)
    """
    if not issubclass(model_class, BaseModel):
        raise TypeError(f"model_class doit hériter de BaseModel. " f"Reçu: {type(model_class)}")

    MODEL_REGISTRY[name.lower()] = model_class


def is_model_available(name: str) -> bool:
    """
    Vérifie si un modèle est disponible.

    Args:
        name: Nom du modèle

    Returns:
        True si le modèle est disponible, False sinon

    Example:
        >>> is_model_available("catboost")
        True  # Si CatBoost est installé
    """
    return name.lower() in MODEL_REGISTRY


# Modèles recommandés pour différents cas d'usage
FAST_MODELS = ["decisiontree", "logistic"]  # Pour landmarking, screening rapide
DEFAULT_MODELS = ["xgboost", "lightgbm", "randomforest"]  # Bon équilibre performance/temps
ALL_TREE_MODELS = ["xgboost", "lightgbm", "randomforest", "decisiontree"]
if CATBOOST_AVAILABLE:
    DEFAULT_MODELS.append("catboost")
    ALL_TREE_MODELS.append("catboost")
