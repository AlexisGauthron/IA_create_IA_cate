"""
Wrapper LightGBM avec interface BaseModel.
"""

from typing import Any, Union

import lightgbm as lgb

from src.models.base import BaseModel


class LightGBMModel(BaseModel):
    """
    Wrapper pour LightGBM avec interface uniforme.

    LightGBM est un algorithme de gradient boosting développé par Microsoft,
    optimisé pour la vitesse et l'efficacité mémoire. Excellent pour les
    grands datasets.

    Example:
        >>> model = LightGBMModel(is_regression=False)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def get_name(self) -> str:
        """Retourne 'lightgbm'."""
        return "lightgbm"

    def get_default_params(self) -> dict[str, Any]:
        """
        Hyperparamètres par défaut pour LightGBM.

        Returns:
            Dictionnaire des paramètres optimisés.
        """
        return {
            "n_estimators": 100,
            "max_depth": -1,  # -1 = pas de limite
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,  # Supprime les logs
            "force_col_wise": True,  # Meilleure performance sur petits datasets
        }

    def get_hp_space(self) -> dict[str, Any]:
        """
        Espace de recherche des hyperparamètres pour HPO.

        Returns:
            Dictionnaire définissant les ranges pour Optuna.
        """
        return {
            "n_estimators": (50, 500),
            "max_depth": (3, 15),
            "learning_rate": (0.01, 0.3),
            "num_leaves": (20, 150),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "min_child_samples": (5, 100),
            "reg_alpha": (0.0, 1.0),
            "reg_lambda": (0.0, 1.0),
        }

    def create_model(self, **hp: Any) -> Union[lgb.LGBMClassifier, lgb.LGBMRegressor]:
        """
        Crée une instance LightGBM.

        Args:
            **hp: Hyperparamètres à passer au constructeur

        Returns:
            LGBMClassifier ou LGBMRegressor selon is_regression
        """
        # Nettoyer les paramètres
        clean_hp = {k: v for k, v in hp.items() if v is not None}

        if self.is_regression:
            return lgb.LGBMRegressor(**clean_hp)
        return lgb.LGBMClassifier(**clean_hp)
