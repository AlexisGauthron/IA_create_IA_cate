"""
Métriques d'évaluation unifiées pour classification et régression.

Ce module centralise toutes les métriques utilisées dans le pipeline :
- LLMFE (évaluation des features)
- Model Selection (screening, HPO)
- Validation finale

Example:
    >>> from src.models.evaluation.metrics import get_metric, get_scorer
    >>> metric_fn = get_metric("f1", is_regression=False)
    >>> score = metric_fn(y_true, y_pred)
    >>> scorer = get_scorer("auc", is_regression=False)
    >>> cv_score = cross_val_score(model, X, y, scoring=scorer)
"""

from typing import Any, Callable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# ============================================================================
# MÉTRIQUES DE CLASSIFICATION
# ============================================================================

CLASSIFICATION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=0),
    "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro", zero_division=0),
    "f1_micro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro", zero_division=0),
    "precision": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "recall": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
}

# Métriques nécessitant des probabilités
CLASSIFICATION_PROBA_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auc": lambda y_true, y_proba: _safe_roc_auc(y_true, y_proba),
    "logloss": lambda y_true, y_proba: log_loss(y_true, y_proba, labels=np.unique(y_true)),
}

# ============================================================================
# MÉTRIQUES DE RÉGRESSION
# ============================================================================

REGRESSION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
    "mape": lambda y_true, y_pred: _safe_mape(y_true, y_pred),
}


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calcule l'AUC de manière sécurisée.

    Gère les cas binaires et multiclasses.

    Args:
        y_true: Labels réels
        y_proba: Probabilités prédites

    Returns:
        Score AUC (0.5 si calcul impossible)
    """
    try:
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            # Cas binaire : utiliser les probabilités de la classe positive
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            return roc_auc_score(y_true, y_proba)

        # Cas multiclasse : utiliser OvR
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")

    except (ValueError, IndexError):
        # Retourner 0.5 (aléatoire) si calcul impossible
        return 0.5


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule le MAPE de manière sécurisée.

    Évite la division par zéro.

    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites

    Returns:
        MAPE en pourcentage
    """
    mask = y_true != 0
    if not mask.any():
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def get_metric(name: str, is_regression: bool = False) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Récupère une fonction de métrique par son nom.

    Args:
        name: Nom de la métrique ('accuracy', 'f1', 'rmse', etc.)
        is_regression: True pour régression, False pour classification

    Returns:
        Fonction callable(y_true, y_pred) -> float

    Raises:
        ValueError: Si la métrique est inconnue

    Example:
        >>> metric = get_metric("f1", is_regression=False)
        >>> score = metric(y_true, y_pred)
    """
    name_lower = name.lower()

    if is_regression:
        if name_lower in REGRESSION_METRICS:
            return REGRESSION_METRICS[name_lower]
        available = list(REGRESSION_METRICS.keys())
        raise ValueError(f"Métrique de régression inconnue: '{name}'. Disponibles: {available}")

    # Classification
    if name_lower in CLASSIFICATION_METRICS:
        return CLASSIFICATION_METRICS[name_lower]
    if name_lower in CLASSIFICATION_PROBA_METRICS:
        return CLASSIFICATION_PROBA_METRICS[name_lower]

    available = list(CLASSIFICATION_METRICS.keys()) + list(CLASSIFICATION_PROBA_METRICS.keys())
    raise ValueError(f"Métrique de classification inconnue: '{name}'. Disponibles: {available}")


def get_scorer(name: str, is_regression: bool = False) -> Any:
    """
    Crée un scorer sklearn compatible avec cross_val_score.

    Args:
        name: Nom de la métrique
        is_regression: True pour régression, False pour classification

    Returns:
        Scorer sklearn

    Example:
        >>> scorer = get_scorer("f1", is_regression=False)
        >>> scores = cross_val_score(model, X, y, scoring=scorer)
    """
    name_lower = name.lower()

    # Métriques où plus grand = meilleur
    greater_is_better = name_lower not in ["logloss", "rmse", "mse", "mae", "mape"]

    if is_regression:
        metric_fn = get_metric(name, is_regression=True)
        return make_scorer(metric_fn, greater_is_better=greater_is_better)

    # Classification avec probabilités
    if name_lower in CLASSIFICATION_PROBA_METRICS:
        metric_fn = CLASSIFICATION_PROBA_METRICS[name_lower]
        return make_scorer(metric_fn, greater_is_better=greater_is_better, needs_proba=True)

    # Classification standard
    metric_fn = get_metric(name, is_regression=False)
    return make_scorer(metric_fn, greater_is_better=greater_is_better)


def get_default_metric(is_regression: bool = False) -> str:
    """
    Retourne la métrique par défaut recommandée.

    Args:
        is_regression: True pour régression, False pour classification

    Returns:
        Nom de la métrique par défaut

    Example:
        >>> get_default_metric(is_regression=False)
        'f1'
        >>> get_default_metric(is_regression=True)
        'rmse'
    """
    return "rmse" if is_regression else "f1"


def list_metrics(is_regression: bool = False) -> list[str]:
    """
    Liste toutes les métriques disponibles.

    Args:
        is_regression: True pour régression, False pour classification

    Returns:
        Liste des noms de métriques

    Example:
        >>> list_metrics(is_regression=False)
        ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'precision', 'recall', 'auc', 'logloss']
    """
    if is_regression:
        return list(REGRESSION_METRICS.keys())
    return list(CLASSIFICATION_METRICS.keys()) + list(CLASSIFICATION_PROBA_METRICS.keys())


def is_proba_metric(name: str) -> bool:
    """
    Vérifie si une métrique nécessite des probabilités.

    Args:
        name: Nom de la métrique

    Returns:
        True si la métrique nécessite predict_proba

    Example:
        >>> is_proba_metric("auc")
        True
        >>> is_proba_metric("accuracy")
        False
    """
    return name.lower() in CLASSIFICATION_PROBA_METRICS


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    is_regression: bool = False,
) -> dict[str, float]:
    """
    Calcule toutes les métriques disponibles.

    Utile pour le reporting complet.

    Args:
        y_true: Labels/valeurs réels
        y_pred: Prédictions
        y_proba: Probabilités (optionnel, pour classification)
        is_regression: True pour régression

    Returns:
        Dictionnaire {nom_métrique: score}

    Example:
        >>> metrics = compute_all_metrics(y_true, y_pred, y_proba)
        >>> print(metrics)
        {'accuracy': 0.85, 'f1': 0.82, 'auc': 0.91, ...}
    """
    results = {}

    if is_regression:
        for name, fn in REGRESSION_METRICS.items():
            try:
                results[name] = fn(y_true, y_pred)
            except Exception:
                results[name] = np.nan
    else:
        # Métriques standard
        for name, fn in CLASSIFICATION_METRICS.items():
            try:
                results[name] = fn(y_true, y_pred)
            except Exception:
                results[name] = np.nan

        # Métriques avec probabilités
        if y_proba is not None:
            for name, fn in CLASSIFICATION_PROBA_METRICS.items():
                try:
                    results[name] = fn(y_true, y_proba)
                except Exception:
                    results[name] = np.nan

    return results
