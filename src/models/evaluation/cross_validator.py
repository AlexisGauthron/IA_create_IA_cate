"""
CrossValidator unifié pour évaluation cohérente des modèles.

Ce module fournit une interface unifiée pour la validation croisée,
utilisable par LLMFE, Model Selection, et HPO.

Example:
    >>> from src.models.evaluation import CrossValidator
    >>> from src.models.registry import get_model
    >>>
    >>> cv = CrossValidator(n_folds=5)
    >>> model = get_model("xgboost", is_regression=False)
    >>> result = cv.evaluate(model, X, y, metric="f1")
    >>> print(f"Score: {result['mean']:.4f} (+/- {result['std']:.4f})")
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.models.base import BaseModel
from src.models.evaluation.metrics import (
    get_default_metric,
    get_metric,
    is_proba_metric,
)


@dataclass
class CVResult:
    """
    Résultat d'une évaluation en validation croisée.

    Attributes:
        mean: Score moyen sur les folds
        std: Écart-type des scores
        scores: Liste des scores par fold
        metric: Nom de la métrique utilisée
        model_name: Nom du modèle évalué
    """

    mean: float
    std: float
    scores: list[float]
    metric: str
    model_name: str

    def __repr__(self) -> str:
        return f"CVResult({self.model_name}, {self.metric}={self.mean:.4f} +/- {self.std:.4f})"

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "mean": self.mean,
            "std": self.std,
            "scores": self.scores,
            "metric": self.metric,
            "model_name": self.model_name,
        }


@dataclass
class MultiModelCVResult:
    """
    Résultat d'une évaluation multi-modèle.

    Attributes:
        results: Résultats par modèle
        aggregated_score: Score agrégé (selon la stratégie)
        aggregation: Stratégie d'agrégation utilisée
        best_model: Nom du meilleur modèle
    """

    results: dict[str, CVResult]
    aggregated_score: float
    aggregation: str
    best_model: str

    def __repr__(self) -> str:
        return (
            f"MultiModelCVResult(aggregated={self.aggregated_score:.4f}, "
            f"best={self.best_model}, n_models={len(self.results)})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "results": {name: r.to_dict() for name, r in self.results.items()},
            "aggregated_score": self.aggregated_score,
            "aggregation": self.aggregation,
            "best_model": self.best_model,
        }


class CrossValidator:
    """
    Validation croisée unifiée pour tous les modèles.

    Supporte :
    - Évaluation d'un modèle unique
    - Évaluation multi-modèle avec agrégation
    - Classification (stratifiée) et régression

    Example:
        >>> cv = CrossValidator(n_folds=5, shuffle=True, random_state=42)
        >>>
        >>> # Évaluation simple
        >>> result = cv.evaluate(model, X, y, metric="f1")
        >>>
        >>> # Évaluation multi-modèle
        >>> models = [get_model("xgboost"), get_model("lightgbm")]
        >>> multi_result = cv.evaluate_multi_model(models, X, y, aggregation="mean")
    """

    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        """
        Initialise le CrossValidator.

        Args:
            n_folds: Nombre de folds (par défaut 5)
            shuffle: Si True, mélange les données avant de créer les folds
            random_state: Graine aléatoire pour reproductibilité
        """
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def _get_kfold(self, is_regression: bool) -> KFold | StratifiedKFold:
        """
        Retourne le splitter approprié.

        Utilise StratifiedKFold pour la classification pour préserver
        la distribution des classes dans chaque fold.

        Args:
            is_regression: True pour régression

        Returns:
            KFold ou StratifiedKFold
        """
        if is_regression:
            return KFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        return StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

    def evaluate(
        self,
        model: BaseModel,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        metric: str = "auto",
    ) -> CVResult:
        """
        Évalue un modèle en validation croisée.

        Args:
            model: Instance de BaseModel à évaluer
            X: Features (DataFrame ou array)
            y: Target
            metric: Nom de la métrique ('auto' pour défaut)

        Returns:
            CVResult avec scores par fold

        Example:
            >>> model = get_model("xgboost", is_regression=False)
            >>> result = cv.evaluate(model, X, y, metric="f1")
            >>> print(result.mean)
        """
        # Déterminer la métrique
        if metric == "auto":
            metric = get_default_metric(model.is_regression)

        metric_fn = get_metric(metric, model.is_regression)
        needs_proba = is_proba_metric(metric)

        # Convertir en numpy si nécessaire
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y)

        # Créer le splitter
        kfold = self._get_kfold(model.is_regression)

        scores = []
        for train_idx, val_idx in kfold.split(X_arr, y_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            # Clone et entraîne le modèle
            model_clone = model.clone()
            model_clone.fit(
                pd.DataFrame(X_train) if isinstance(X, pd.DataFrame) else X_train,
                y_train,
            )

            # Prédiction
            if needs_proba:
                y_pred = model_clone.predict_proba(
                    pd.DataFrame(X_val) if isinstance(X, pd.DataFrame) else X_val
                )
            else:
                y_pred = model_clone.predict(
                    pd.DataFrame(X_val) if isinstance(X, pd.DataFrame) else X_val
                )

            # Calcul du score
            score = metric_fn(y_val, y_pred)
            scores.append(score)

        return CVResult(
            mean=float(np.mean(scores)),
            std=float(np.std(scores)),
            scores=scores,
            metric=metric,
            model_name=model.get_name(),
        )

    def evaluate_multi_model(
        self,
        models: list[BaseModel],
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        metric: str = "auto",
        aggregation: str = "mean",
    ) -> MultiModelCVResult:
        """
        Évalue plusieurs modèles et agrège les scores.

        Utile pour LLMFE où on veut évaluer les features sur plusieurs modèles
        pour éviter l'overfitting à un algorithme spécifique.

        Args:
            models: Liste d'instances BaseModel
            X: Features
            y: Target
            metric: Nom de la métrique
            aggregation: Stratégie d'agrégation ('mean', 'min', 'max', 'median')

        Returns:
            MultiModelCVResult avec résultats par modèle et score agrégé

        Example:
            >>> models = [get_model("xgboost"), get_model("lightgbm"), get_model("randomforest")]
            >>> result = cv.evaluate_multi_model(models, X, y, aggregation="mean")
            >>> print(f"Score agrégé: {result.aggregated_score:.4f}")
            >>> print(f"Meilleur modèle: {result.best_model}")
        """
        if not models:
            raise ValueError("La liste de modèles ne peut pas être vide")

        # Évaluer chaque modèle
        results: dict[str, CVResult] = {}
        for model in models:
            result = self.evaluate(model, X, y, metric=metric)
            results[model.get_name()] = result

        # Extraire les scores moyens
        mean_scores = [r.mean for r in results.values()]

        # Agrégation
        aggregation_fns = {
            "mean": np.mean,
            "min": np.min,
            "max": np.max,
            "median": np.median,
        }

        if aggregation not in aggregation_fns:
            raise ValueError(
                f"Agrégation inconnue: '{aggregation}'. "
                f"Disponibles: {list(aggregation_fns.keys())}"
            )

        aggregated_score = float(aggregation_fns[aggregation](mean_scores))

        # Trouver le meilleur modèle
        best_model = max(results.items(), key=lambda x: x[1].mean)[0]

        return MultiModelCVResult(
            results=results,
            aggregated_score=aggregated_score,
            aggregation=aggregation,
            best_model=best_model,
        )

    def quick_evaluate(
        self,
        model: BaseModel,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        metric: str = "auto",
        sample_size: int | None = None,
    ) -> float:
        """
        Évaluation rapide avec un seul fold ou échantillon.

        Utile pour le screening initial ou le landmarking.

        Args:
            model: Modèle à évaluer
            X: Features
            y: Target
            metric: Métrique
            sample_size: Taille d'échantillon (None = tout)

        Returns:
            Score unique (approximatif mais rapide)

        Example:
            >>> score = cv.quick_evaluate(model, X, y, sample_size=1000)
        """
        # Sous-échantillonnage si demandé
        if sample_size is not None and len(y) > sample_size:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(y), size=sample_size, replace=False)
            if isinstance(X, pd.DataFrame):
                X = X.iloc[idx]
            else:
                X = X[idx]
            y = y[idx]

        # Évaluation avec 1 fold (train/val split simple)
        cv_1fold = CrossValidator(n_folds=2, shuffle=True, random_state=self.random_state)
        result = cv_1fold.evaluate(model, X, y, metric=metric)

        return result.scores[0]  # Premier fold seulement
