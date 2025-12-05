# src/feature_engineering/dfs/selection.py
"""
Sélection automatique des features générées par DFS.

DFS peut générer des centaines de features. Ce module permet de :
- Filtrer les features redondantes (corrélation élevée)
- Sélectionner les features importantes (basé sur les modèles)
- Appliquer RFE (Recursive Feature Elimination)
- Approche hybride combinant plusieurs méthodes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder


@dataclass
class SelectionResult:
    """Résultat de la sélection de features."""

    selected_features: list[str]
    dropped_features: list[str]
    feature_scores: dict[str, float]
    method_used: str
    n_original: int
    n_selected: int

    @property
    def reduction_pct(self) -> float:
        """Pourcentage de réduction."""
        if self.n_original == 0:
            return 0.0
        return (1 - self.n_selected / self.n_original) * 100


class FeatureSelector:
    """
    Sélecteur de features pour réduire l'ensemble généré par DFS.

    Méthodes disponibles:
    - importance: Basé sur l'importance des features (RF/XGBoost)
    - correlation: Supprime les features trop corrélées entre elles
    - rfe: Recursive Feature Elimination
    - hybrid: Combinaison de correlation + importance
    """

    def __init__(
        self,
        method: str = "importance",
        threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        top_k: int | None = None,
        is_regression: bool = False,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Initialise le sélecteur.

        Args:
            method: Méthode de sélection ('importance', 'correlation', 'rfe', 'hybrid')
            threshold: Seuil d'importance minimale (pour 'importance')
            correlation_threshold: Seuil de corrélation max entre features (pour 'correlation')
            top_k: Garder uniquement les top K features (optionnel)
            is_regression: True si régression, False si classification
            random_state: Graine aléatoire
            verbose: Afficher les logs
        """
        self.method = method
        self.threshold = threshold
        self.correlation_threshold = correlation_threshold
        self.top_k = top_k
        self.is_regression = is_regression
        self.random_state = random_state
        self.verbose = verbose

        self._feature_importances: dict[str, float] = {}
        self._correlation_matrix: pd.DataFrame | None = None

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> tuple[pd.DataFrame, SelectionResult]:
        """
        Sélectionne les features et retourne le DataFrame réduit.

        Args:
            X: DataFrame avec toutes les features DFS
            y: Target (pour calcul d'importance)

        Returns:
            Tuple (DataFrame réduit, SelectionResult)
        """
        if self.verbose:
            print(f"\n[SELECTION] Méthode: {self.method}")
            print(f"[SELECTION] Features initiales: {X.shape[1]}")

        # Pré-traitement: supprimer colonnes constantes et avec trop de NaN
        X_clean = self._preprocess(X)

        if self.method == "importance":
            selected, scores = self._select_by_importance(X_clean, y)
        elif self.method == "correlation":
            selected, scores = self._select_by_correlation(X_clean, y)
        elif self.method == "rfe":
            selected, scores = self._select_by_rfe(X_clean, y)
        elif self.method == "hybrid":
            selected, scores = self._select_hybrid(X_clean, y)
        else:
            raise ValueError(f"Méthode inconnue: {self.method}")

        # Appliquer top_k si spécifié
        if self.top_k is not None and len(selected) > self.top_k:
            # Trier par score et garder les top K
            sorted_features = sorted(selected, key=lambda f: scores.get(f, 0), reverse=True)
            selected = sorted_features[: self.top_k]

        dropped = [col for col in X_clean.columns if col not in selected]

        result = SelectionResult(
            selected_features=selected,
            dropped_features=dropped,
            feature_scores=scores,
            method_used=self.method,
            n_original=X.shape[1],
            n_selected=len(selected),
        )

        if self.verbose:
            print(f"[SELECTION] Features sélectionnées: {result.n_selected}")
            print(f"[SELECTION] Réduction: {result.reduction_pct:.1f}%")

        return X[selected], result

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Pré-traitement: supprime colonnes problématiques."""
        X_clean = X.copy()

        # Supprimer colonnes avec trop de NaN (>50%)
        nan_ratio = X_clean.isna().mean()
        cols_to_drop = nan_ratio[nan_ratio > 0.5].index.tolist()
        if cols_to_drop and self.verbose:
            print(f"[SELECTION] Suppression {len(cols_to_drop)} colonnes avec >50% NaN")
        X_clean = X_clean.drop(columns=cols_to_drop)

        # Supprimer colonnes constantes
        nunique = X_clean.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols and self.verbose:
            print(f"[SELECTION] Suppression {len(constant_cols)} colonnes constantes")
        X_clean = X_clean.drop(columns=constant_cols)

        # Remplir les NaN restants
        for col in X_clean.columns:
            if X_clean[col].isna().any():
                if X_clean[col].dtype.name == "category":
                    # Pour les colonnes category, ajouter d'abord la catégorie
                    if "MISSING" not in X_clean[col].cat.categories:
                        X_clean[col] = X_clean[col].cat.add_categories(["MISSING"])
                    X_clean[col] = X_clean[col].fillna("MISSING")
                elif X_clean[col].dtype == "object":
                    X_clean[col] = X_clean[col].fillna("MISSING")
                else:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())

        # Encoder les colonnes catégorielles
        for col in X_clean.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X_clean[col] = le.fit_transform(X_clean[col].astype(str))

        return X_clean

    def _select_by_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> tuple[list[str], dict[str, float]]:
        """Sélection basée sur l'importance des features."""
        if self.verbose:
            print("[SELECTION] Calcul des importances via RandomForest...")

        # Utiliser RandomForest pour calculer les importances
        if self.is_regression:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            )

        # Encoder la target si nécessaire
        y_encoded = y
        if not self.is_regression and hasattr(y, "dtype") and y.dtype == "object":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

        model.fit(X, y_encoded)

        # Récupérer les importances
        importances = dict(zip(X.columns, model.feature_importances_, strict=False))
        self._feature_importances = importances

        # Filtrer par seuil
        selected = [col for col, imp in importances.items() if imp >= self.threshold]

        if self.verbose:
            print(f"[SELECTION] Features avec importance >= {self.threshold}: {len(selected)}")

        return selected, importances

    def _select_by_correlation(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> tuple[list[str], dict[str, float]]:
        """Sélection basée sur la corrélation (supprime les redondantes)."""
        if self.verbose:
            print("[SELECTION] Calcul de la matrice de corrélation...")

        # Calculer la matrice de corrélation
        corr_matrix = X.corr().abs()
        self._correlation_matrix = corr_matrix

        # Calculer la corrélation avec la target pour le score
        y_encoded = y
        if hasattr(y, "dtype") and y.dtype == "object":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

        target_corr = {}
        for col in X.columns:
            try:
                target_corr[col] = abs(np.corrcoef(X[col].values, y_encoded)[0, 1])
            except Exception:
                target_corr[col] = 0.0

        # Identifier les features à supprimer (trop corrélées entre elles)
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper_tri.columns:
            # Features fortement corrélées avec cette colonne
            correlated = upper_tri.index[upper_tri[col] > self.correlation_threshold].tolist()
            for corr_col in correlated:
                # Garder celle avec la meilleure corrélation avec la target
                if target_corr.get(col, 0) >= target_corr.get(corr_col, 0):
                    to_drop.add(corr_col)
                else:
                    to_drop.add(col)

        selected = [col for col in X.columns if col not in to_drop]

        if self.verbose:
            print(
                f"[SELECTION] Features supprimées (corrélation > {self.correlation_threshold}): {len(to_drop)}"
            )

        return selected, target_corr

    def _select_by_rfe(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        n_features_to_select: int | None = None,
    ) -> tuple[list[str], dict[str, float]]:
        """Recursive Feature Elimination."""
        if self.verbose:
            print("[SELECTION] Application de RFE...")

        if n_features_to_select is None:
            n_features_to_select = self.top_k or max(10, X.shape[1] // 4)

        # Estimateur de base
        if self.is_regression:
            estimator = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1,
            )

        # Encoder la target si nécessaire
        y_encoded = y
        if not self.is_regression and hasattr(y, "dtype") and y.dtype == "object":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

        # RFE
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=0.1,  # Supprimer 10% des features à chaque étape
        )
        selector.fit(X, y_encoded)

        # Récupérer les features sélectionnées
        selected = X.columns[selector.support_].tolist()

        # Scores basés sur le ranking (inverse)
        scores = dict(zip(X.columns, 1 / selector.ranking_, strict=False))

        if self.verbose:
            print(f"[SELECTION] Features sélectionnées par RFE: {len(selected)}")

        return selected, scores

    def _select_hybrid(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> tuple[list[str], dict[str, float]]:
        """
        Approche hybride:
        1. D'abord supprimer les features trop corrélées
        2. Ensuite sélectionner par importance
        """
        if self.verbose:
            print("[SELECTION] Approche hybride: correlation + importance")

        # Étape 1: Filtrage par corrélation
        corr_selected, corr_scores = self._select_by_correlation(X, y)
        X_filtered = X[corr_selected]

        if self.verbose:
            print(f"[SELECTION] Après filtrage corrélation: {len(corr_selected)} features")

        # Étape 2: Sélection par importance
        imp_selected, imp_scores = self._select_by_importance(X_filtered, y)

        # Combiner les scores (normaliser et moyenner)
        combined_scores = {}
        for col in imp_selected:
            corr_score = corr_scores.get(col, 0)
            imp_score = imp_scores.get(col, 0)
            combined_scores[col] = (corr_score + imp_score) / 2

        return imp_selected, combined_scores

    def get_feature_importances(self) -> dict[str, float]:
        """Retourne les importances calculées."""
        return self._feature_importances

    def get_correlation_matrix(self) -> pd.DataFrame | None:
        """Retourne la matrice de corrélation calculée."""
        return self._correlation_matrix


def select_features(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    method: str = "hybrid",
    threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    top_k: int | None = None,
    is_regression: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, SelectionResult]:
    """
    Fonction raccourcie pour sélectionner les features.

    Args:
        X: DataFrame avec les features
        y: Target
        method: Méthode de sélection
        threshold: Seuil d'importance
        correlation_threshold: Seuil de corrélation
        top_k: Nombre max de features à garder
        is_regression: Type de tâche
        verbose: Afficher les logs

    Returns:
        Tuple (DataFrame filtré, SelectionResult)
    """
    selector = FeatureSelector(
        method=method,
        threshold=threshold,
        correlation_threshold=correlation_threshold,
        top_k=top_k,
        is_regression=is_regression,
        verbose=verbose,
    )
    return selector.fit_transform(X, y)
