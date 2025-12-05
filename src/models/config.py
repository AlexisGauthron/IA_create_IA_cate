"""
Configuration centralisée pour le module modèles.

Ce module définit les configurations par défaut et les presets
pour différents cas d'usage (screening rapide, HPO complet, etc.).

Example:
    >>> from src.models.config import ModelConfig, SCREENING_CONFIG, HPO_CONFIG
    >>> config = ModelConfig()
    >>> config.models  # ['xgboost', 'lightgbm', 'randomforest']
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    """
    Configuration pour l'évaluation des modèles.

    Attributes:
        models: Liste des modèles à utiliser
        metric: Métrique d'évaluation ('auto' pour défaut)
        n_folds: Nombre de folds pour la validation croisée
        random_state: Graine aléatoire
        n_jobs: Nombre de jobs parallèles (-1 = tous les CPU)
        timeout_per_model: Timeout en secondes par modèle (None = pas de limite)
    """

    models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm", "randomforest"])
    metric: str = "auto"
    n_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    timeout_per_model: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "models": self.models,
            "metric": self.metric,
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "timeout_per_model": self.timeout_per_model,
        }


@dataclass
class ScreeningConfig(ModelConfig):
    """
    Configuration pour le screening rapide des modèles.

    Utilise des modèles simples et peu de folds pour évaluer rapidement
    quelle stratégie fonctionne le mieux.
    """

    models: list[str] = field(default_factory=lambda: ["decisiontree", "logistic"])
    n_folds: int = 3
    timeout_per_model: int = 60  # 1 minute max par modèle


@dataclass
class DefaultConfig(ModelConfig):
    """
    Configuration par défaut équilibrée.

    Bon compromis entre performance et temps d'exécution.
    """

    models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm", "randomforest"])
    n_folds: int = 5
    timeout_per_model: int = 300  # 5 minutes max par modèle


@dataclass
class HPOConfig(ModelConfig):
    """
    Configuration pour l'optimisation d'hyperparamètres.

    Utilise plus de folds pour une évaluation plus robuste.
    """

    models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    n_folds: int = 5
    n_trials: int = 50  # Nombre de trials Optuna
    timeout_total: int = 3600  # 1 heure max au total


@dataclass
class LLMFEConfig(ModelConfig):
    """
    Configuration pour l'évaluation des features (LLMFE).

    Utilise plusieurs modèles pour éviter l'overfitting à un algorithme.
    """

    models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm", "randomforest"])
    n_folds: int = 4
    aggregation: str = "mean"  # Comment agréger les scores multi-modèle


@dataclass
class LandmarkingConfig(ModelConfig):
    """
    Configuration pour le landmarking (méta-learning).

    Utilise des modèles très rapides pour extraire des méta-features
    basées sur la performance.
    """

    models: list[str] = field(default_factory=lambda: ["decisiontree", "logistic"])
    n_folds: int = 2
    sample_size: int = 1000  # Sous-échantillonnage pour rapidité


# Presets prédéfinis
SCREENING_CONFIG = ScreeningConfig()
DEFAULT_CONFIG = DefaultConfig()
HPO_CONFIG = HPOConfig()
LLMFE_CONFIG = LLMFEConfig()
LANDMARKING_CONFIG = LandmarkingConfig()


# Mapping des presets par nom
CONFIG_PRESETS: dict[str, ModelConfig] = {
    "screening": SCREENING_CONFIG,
    "default": DEFAULT_CONFIG,
    "hpo": HPO_CONFIG,
    "llmfe": LLMFE_CONFIG,
    "landmarking": LANDMARKING_CONFIG,
}


def get_config(preset: str = "default") -> ModelConfig:
    """
    Récupère une configuration par son nom.

    Args:
        preset: Nom du preset ('screening', 'default', 'hpo', 'llmfe', 'landmarking')

    Returns:
        Instance de configuration

    Raises:
        ValueError: Si le preset est inconnu

    Example:
        >>> config = get_config("llmfe")
        >>> print(config.models)
        ['xgboost', 'lightgbm', 'randomforest']
    """
    if preset not in CONFIG_PRESETS:
        available = list(CONFIG_PRESETS.keys())
        raise ValueError(f"Preset inconnu: '{preset}'. Disponibles: {available}")

    return CONFIG_PRESETS[preset]


def list_presets() -> list[str]:
    """
    Liste les noms de tous les presets disponibles.

    Returns:
        Liste des noms de presets
    """
    return list(CONFIG_PRESETS.keys())
