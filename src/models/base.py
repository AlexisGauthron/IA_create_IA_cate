"""
Module de base pour les modèles ML.

Ce module définit l'interface commune BaseModel que tous les wrappers de modèles
doivent implémenter. Cela permet une utilisation uniforme des modèles dans tout
le pipeline (LLMFE, CASH, screening, HPO).
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Interface commune pour tous les modèles ML.

    Cette classe abstraite définit le contrat que tous les wrappers de modèles
    doivent respecter. Elle permet de switcher facilement entre différents
    algorithmes (XGBoost, LightGBM, RandomForest, etc.) avec une API uniforme.

    Attributes:
        is_regression: True si le problème est une régression, False pour classification
        random_state: Seed pour la reproductibilité
        params: Hyperparamètres additionnels passés au modèle
        model: Instance du modèle sous-jacent (après fit)

    Example:
        >>> model = XGBoostModel(is_regression=False)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        is_regression: bool = False,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialise le modèle.

        Args:
            is_regression: True pour régression, False pour classification
            random_state: Seed pour la reproductibilité
            **kwargs: Hyperparamètres additionnels à passer au modèle
        """
        self.is_regression = is_regression
        self.random_state = random_state
        self.params = kwargs
        self.model: Any = None

    @abstractmethod
    def get_name(self) -> str:
        """
        Retourne le nom du modèle.

        Returns:
            Nom unique du modèle (ex: 'xgboost', 'lightgbm', 'randomforest')
        """
        pass

    @abstractmethod
    def create_model(self, **hp: Any) -> Any:
        """
        Crée une instance du modèle sous-jacent avec les hyperparamètres donnés.

        Args:
            **hp: Hyperparamètres à passer au constructeur du modèle

        Returns:
            Instance du modèle (sklearn-compatible avec fit/predict)
        """
        pass

    @abstractmethod
    def get_default_params(self) -> dict[str, Any]:
        """
        Retourne les hyperparamètres par défaut du modèle.

        Ces paramètres sont utilisés si aucun paramètre n'est spécifié.

        Returns:
            Dictionnaire des hyperparamètres par défaut
        """
        pass

    @abstractmethod
    def get_hp_space(self) -> dict[str, Any]:
        """
        Retourne l'espace de recherche des hyperparamètres pour HPO.

        Utilisé par Optuna ou d'autres optimiseurs pour définir les ranges.

        Returns:
            Dictionnaire définissant l'espace de recherche.
            Format: {"param_name": (min, max)} pour continu
                   {"param_name": [val1, val2, ...]} pour catégoriel
        """
        pass

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseModel":
        """
        Entraîne le modèle sur les données.

        Args:
            X: Features d'entraînement (DataFrame)
            y: Target d'entraînement (array)

        Returns:
            self pour permettre le chaînage
        """
        # Fusion des paramètres par défaut avec ceux passés au constructeur
        params = {**self.get_default_params(), **self.params}
        self.model = self.create_model(**params)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédit les valeurs/classes pour les données.

        Args:
            X: Features à prédire (DataFrame)

        Returns:
            Prédictions (array)

        Raises:
            ValueError: Si le modèle n'a pas été entraîné
        """
        if self.model is None:
            raise ValueError(f"{self.get_name()} n'a pas été entraîné. Appelez fit() d'abord.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédit les probabilités pour chaque classe (classification seulement).

        Args:
            X: Features à prédire (DataFrame)

        Returns:
            Probabilités pour chaque classe (array shape [n_samples, n_classes])

        Raises:
            ValueError: Si le modèle n'a pas été entraîné
            NotImplementedError: Si le modèle ne supporte pas predict_proba
        """
        if self.model is None:
            raise ValueError(f"{self.get_name()} n'a pas été entraîné. Appelez fit() d'abord.")

        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(
                f"{self.get_name()} ne supporte pas predict_proba. "
                "Utilisez predict() pour les prédictions directes."
            )

        return self.model.predict_proba(X)

    def clone(self) -> "BaseModel":
        """
        Crée une copie non entraînée du modèle avec les mêmes paramètres.

        Utile pour la cross-validation où on a besoin de plusieurs instances.

        Returns:
            Nouvelle instance du modèle (non entraînée)
        """
        return self.__class__(
            is_regression=self.is_regression,
            random_state=self.random_state,
            **self.params,
        )

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Retourne l'importance des features si disponible.

        Returns:
            Array des importances ou None si non disponible
        """
        if self.model is None:
            return None

        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_

        return None

    def __repr__(self) -> str:
        """Représentation string du modèle."""
        status = "entraîné" if self.model is not None else "non entraîné"
        task = "régression" if self.is_regression else "classification"
        return f"{self.__class__.__name__}(task={task}, status={status})"
