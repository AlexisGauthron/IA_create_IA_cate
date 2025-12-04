"""
Évaluateur multi-modèle pour LLMFE.

Ce module fournit une fonction d'évaluation des features qui utilise
le module centralisé src/models/ au lieu d'un modèle XGBoost codé en dur.

Avantages:
- Évaluation sur plusieurs modèles (évite l'overfitting à un algo)
- Configuration flexible des modèles
- Métriques standardisées via src/models/evaluation/

Example:
    >>> from src.feature_engineering.llmfe.model_evaluator import evaluate_features
    >>> score = evaluate_features(
    ...     X, y,
    ...     is_regression=False,
    ...     model_names=["xgboost", "lightgbm", "randomforest"],
    ...     n_folds=4,
    ...     aggregation="mean"
    ... )
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold

from src.models import CrossValidator, get_model, get_models


def evaluate_features(
    X: pd.DataFrame,
    y: np.ndarray,
    is_regression: bool = False,
    model_names: list[str] | None = None,
    n_folds: int = 4,
    metric: str = "auto",
    aggregation: str = "mean",
    random_state: int = 42,
) -> float:
    """
    Évalue la qualité des features transformées avec plusieurs modèles.

    Cette fonction remplace l'évaluation XGBoost codée en dur dans LLMFE.
    Elle permet d'évaluer les features sur plusieurs modèles pour éviter
    l'overfitting à un algorithme spécifique.

    Args:
        X: DataFrame des features transformées
        y: Array des labels/valeurs cibles
        is_regression: True pour régression, False pour classification
        model_names: Liste des modèles à utiliser (défaut: ["xgboost"])
        n_folds: Nombre de folds pour la validation croisée
        metric: Métrique d'évaluation ('auto' pour défaut selon le type)
        aggregation: Stratégie d'agrégation multi-modèle ('mean', 'min', 'max')
        random_state: Graine aléatoire pour reproductibilité

    Returns:
        Score agrégé (plus grand = meilleur)

    Example:
        >>> # Évaluation simple (comme avant, avec XGBoost seul)
        >>> score = evaluate_features(X, y, is_regression=False)
        >>>
        >>> # Évaluation multi-modèle (recommandé)
        >>> score = evaluate_features(
        ...     X, y,
        ...     model_names=["xgboost", "lightgbm", "randomforest"],
        ...     aggregation="mean"
        ... )
    """
    # Valeurs par défaut
    if model_names is None:
        model_names = ["xgboost"]

    # Préparer les données
    X_prepared = _prepare_features(X)
    y_prepared = _prepare_target(y, is_regression)

    # Créer le CrossValidator
    cv = CrossValidator(n_folds=n_folds, shuffle=True, random_state=random_state)

    # Évaluation
    if len(model_names) == 1:
        # Un seul modèle : évaluation simple
        model = get_model(model_names[0], is_regression=is_regression)
        result = cv.evaluate(model, X_prepared, y_prepared, metric=metric)
        return result.mean
    else:
        # Plusieurs modèles : évaluation multi-modèle avec agrégation
        models = get_models(model_names, is_regression=is_regression)
        result = cv.evaluate_multi_model(
            models, X_prepared, y_prepared, metric=metric, aggregation=aggregation
        )
        return result.aggregated_score


def evaluate_features_detailed(
    X: pd.DataFrame,
    y: np.ndarray,
    is_regression: bool = False,
    model_names: list[str] | None = None,
    n_folds: int = 4,
    metric: str = "auto",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Évalue les features avec détails complets par modèle.

    Comme evaluate_features() mais retourne les résultats détaillés
    pour chaque modèle au lieu d'un score agrégé.

    Args:
        X: DataFrame des features
        y: Labels/valeurs cibles
        is_regression: True pour régression
        model_names: Liste des modèles
        n_folds: Nombre de folds
        metric: Métrique
        random_state: Graine aléatoire

    Returns:
        Dictionnaire avec:
        - 'scores': {model_name: mean_score}
        - 'std': {model_name: std_score}
        - 'best_model': nom du meilleur modèle
        - 'best_score': score du meilleur modèle

    Example:
        >>> result = evaluate_features_detailed(X, y, model_names=["xgboost", "lightgbm"])
        >>> print(f"Meilleur: {result['best_model']} avec {result['best_score']:.4f}")
    """
    if model_names is None:
        model_names = ["xgboost", "lightgbm", "randomforest"]

    # Préparer les données
    X_prepared = _prepare_features(X)
    y_prepared = _prepare_target(y, is_regression)

    # Créer le CrossValidator
    cv = CrossValidator(n_folds=n_folds, shuffle=True, random_state=random_state)

    # Évaluer chaque modèle
    models = get_models(model_names, is_regression=is_regression)
    multi_result = cv.evaluate_multi_model(models, X_prepared, y_prepared, metric=metric)

    return {
        "scores": {name: res.mean for name, res in multi_result.results.items()},
        "std": {name: res.std for name, res in multi_result.results.items()},
        "best_model": multi_result.best_model,
        "best_score": multi_result.results[multi_result.best_model].mean,
        "metric": metric if metric != "auto" else multi_result.results[model_names[0]].metric,
    }


def _prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les features pour l'évaluation.

    - Encode les colonnes catégorielles string en numérique
    - Remplace les NaN/Inf par des valeurs valides

    Args:
        X: DataFrame des features

    Returns:
        DataFrame préparé
    """
    X_copy = X.copy()
    label_encoder = preprocessing.LabelEncoder()

    for col in X_copy.columns:
        # Encoder les colonnes string/object
        if X_copy[col].dtype == "string" or X_copy[col].dtype == "object":
            # Gérer les valeurs manquantes
            X_copy[col] = X_copy[col].fillna("__MISSING__")
            X_copy[col] = label_encoder.fit_transform(X_copy[col].astype(str))

        # Remplacer inf par NaN puis imputer
        X_copy[col] = X_copy[col].replace([np.inf, -np.inf], np.nan)

    # Imputation simple des NaN restants
    X_copy = X_copy.fillna(X_copy.median())

    # Si encore des NaN (colonnes entièrement vides), remplacer par 0
    X_copy = X_copy.fillna(0)

    return X_copy


def _prepare_target(y: np.ndarray, is_regression: bool) -> np.ndarray:
    """
    Prépare la cible pour l'évaluation.

    - Pour classification : encode les labels si nécessaire
    - Pour régression : retourne tel quel

    Args:
        y: Array des valeurs cibles
        is_regression: True pour régression

    Returns:
        Array préparé
    """
    if is_regression:
        return np.asarray(y, dtype=float)

    # Classification : encoder si nécessaire
    y_array = np.asarray(y)

    if y_array.dtype == object or y_array.dtype.kind in ("U", "S"):
        label_encoder = preprocessing.LabelEncoder()
        return label_encoder.fit_transform(y_array)

    return y_array


# ============================================================================
# FONCTION LEGACY POUR RÉTROCOMPATIBILITÉ
# ============================================================================


def evaluate_with_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    is_regression: bool = False,
    n_folds: int = 4,
    random_state: int = 42,
) -> float:
    """
    Évalue avec XGBoost uniquement (rétrocompatibilité).

    Cette fonction reproduit le comportement original de LLMFE.
    Préférer evaluate_features() pour les nouvelles utilisations.

    Args:
        X: Features
        y: Cibles
        is_regression: Type de problème
        n_folds: Nombre de folds
        random_state: Graine aléatoire

    Returns:
        Score moyen sur les folds
    """
    return evaluate_features(
        X=X,
        y=y,
        is_regression=is_regression,
        model_names=["xgboost"],
        n_folds=n_folds,
        random_state=random_state,
    )
