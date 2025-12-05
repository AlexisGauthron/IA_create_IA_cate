"""Configuration of a llmfe experiments.

Ce module contient les configurations pour LLMFE :
- ExperienceBufferConfig : paramètres du buffer d'expérience
- EvaluationConfig : configuration des modèles d'évaluation (NOUVEAU)
- Config : configuration principale
- ClassConfig : classes de sampler et sandbox
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from src.feature_engineering.llmfe import evaluator, sampler


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    """Configuration des modèles d'évaluation pour LLMFE.

    Permet de configurer quels modèles utiliser pour évaluer les features
    générées, au lieu d'utiliser uniquement XGBoost.

    Supporte deux modes :
    1. Métrique unique (legacy) : `metric='f1'`
    2. Métriques pondérées (nouveau) : `metrics_config=[{"name": "recall", "weight": 0.6}, ...]`

    Args:
        model_names: Liste des modèles à utiliser pour l'évaluation.
            Options: 'xgboost', 'lightgbm', 'randomforest', 'decisiontree', 'logistic'
            Par défaut: ['xgboost'] (comportement legacy)
        n_folds: Nombre de folds pour la validation croisée
        metric: Métrique d'évaluation unique ('auto', 'f1', 'accuracy', 'auc', 'rmse', etc.)
        metrics_config: Configuration multi-métrique pondérée (prioritaire sur `metric`)
            Format: [{"name": "recall", "weight": 0.5}, {"name": "precision", "weight": 0.5}]
            La somme des poids doit être 1.0
        aggregation: Stratégie d'agrégation multi-modèle ('mean', 'min', 'max')
        use_multi_model: Si True, utilise plusieurs modèles (override model_names)

    Example:
        >>> # Configuration legacy (XGBoost seul, métrique unique)
        >>> eval_config = EvaluationConfig()
        >>>
        >>> # Configuration multi-modèle avec métrique unique
        >>> eval_config = EvaluationConfig(
        ...     model_names=['xgboost', 'lightgbm', 'randomforest'],
        ...     metric='f1',
        ...     aggregation='mean'
        ... )
        >>>
        >>> # NOUVEAU : Configuration avec métriques pondérées
        >>> eval_config = EvaluationConfig(
        ...     model_names=['xgboost'],
        ...     metrics_config=[
        ...         {"name": "recall", "weight": 0.6},
        ...         {"name": "precision", "weight": 0.3},
        ...         {"name": "f1", "weight": 0.1}
        ...     ]
        ... )
    """

    model_names: tuple[str, ...] = ("xgboost",)
    n_folds: int = 4
    metric: str = "auto"
    metrics_config: tuple[dict, ...] | None = None  # NOUVEAU : métriques pondérées
    aggregation: Literal["mean", "min", "max", "median"] = "mean"
    use_multi_model: bool = False

    def get_model_names(self) -> list[str]:
        """Retourne la liste des modèles à utiliser."""
        if self.use_multi_model and self.model_names == ("xgboost",):
            # Si multi-modèle activé mais liste par défaut, utiliser les 3 principaux
            return ["xgboost", "lightgbm", "randomforest"]
        return list(self.model_names)

    def get_metrics_config(self) -> list[dict] | None:
        """Retourne la configuration des métriques pondérées (ou None si mono-métrique)."""
        if self.metrics_config:
            return list(self.metrics_config)
        return None

    def is_weighted_metrics(self) -> bool:
        """Retourne True si utilise des métriques pondérées."""
        return self.metrics_config is not None and len(self.metrics_config) > 0


# Presets de configuration d'évaluation
EVAL_LEGACY = EvaluationConfig()  # XGBoost seul (comportement original)
EVAL_MULTI_MODEL = EvaluationConfig(
    model_names=("xgboost", "lightgbm", "randomforest"),
    aggregation="mean",
)
EVAL_FAST = EvaluationConfig(
    model_names=("decisiontree", "logistic"),
    n_folds=3,
    aggregation="mean",
)


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters.

    Args:
        functions_per_prompt (int): Number of previous hypotheses to include in prompts
        num_islands (int): Number of islands in experience buffer for diversity
        reset_period (int): Seconds between weakest island resets
        cluster_sampling_temperature_init (float): Initial cluster softmax sampling temperature
        cluster_sampling_temperature_period (int): Period for temperature decay
    """

    functions_per_prompt: int = 2
    num_islands: int = 3
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for llmfe experiments.

    Args:
        experience_buffer: Evolution multi-population settings
        evaluation: Configuration des modèles d'évaluation (NOUVEAU)
        num_samplers (int): Number of parallel samplers
        num_evaluators (int): Number of parallel evaluators
        samples_per_prompt (int): Number of hypotheses per prompt
        evaluate_timeout_seconds (int): Hypothesis evaluation timeout
        use_api (bool): API usage flag
        api_model (str): Modèle API à utiliser

    Example:
        >>> # Configuration avec évaluation multi-modèle
        >>> config = Config(
        ...     evaluation=EvaluationConfig(
        ...         model_names=('xgboost', 'lightgbm'),
        ...         aggregation='mean'
        ...     ),
        ...     use_api=True,
        ...     api_model='gpt-4'
        ... )
    """

    experience_buffer: ExperienceBufferConfig = dataclasses.field(
        default_factory=ExperienceBufferConfig
    )
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)
    num_samplers: int = 1
    num_evaluators: int = 1
    samples_per_prompt: int = 3
    evaluate_timeout_seconds: int = 30
    use_api: bool = False
    api_model: str = "gpt-3.5-turbo"


@dataclasses.dataclass()
class ClassConfig:
    llm_class: type[sampler.LLM]
    sandbox_class: type[evaluator.Sandbox]
