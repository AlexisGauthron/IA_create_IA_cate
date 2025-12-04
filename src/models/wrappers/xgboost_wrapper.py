"""
Wrapper XGBoost avec interface BaseModel.
"""

from typing import Any

import xgboost as xgb

from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    """
    Wrapper pour XGBoost avec interface uniforme.

    XGBoost est un algorithme de gradient boosting optimisé, très performant
    pour les données tabulaires. Il supporte classification et régression.

    Example:
        >>> model = XGBoostModel(is_regression=False)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> probas = model.predict_proba(X_test)
    """

    def get_name(self) -> str:
        """Retourne 'xgboost'."""
        return "xgboost"

    def get_default_params(self) -> dict[str, Any]:
        """
        Hyperparamètres par défaut pour XGBoost.

        Returns:
            Dictionnaire des paramètres optimisés pour un bon compromis
            performance/temps d'entraînement.
        """
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
            # Éviter les warnings pour les nouvelles versions
            "use_label_encoder": False,
            "eval_metric": "logloss" if not self.is_regression else "rmse",
        }

    def get_hp_space(self) -> dict[str, Any]:
        """
        Espace de recherche des hyperparamètres pour HPO.

        Returns:
            Dictionnaire définissant les ranges pour Optuna/autres optimiseurs.
        """
        return {
            "n_estimators": (50, 500),
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "min_child_weight": (1, 10),
            "gamma": (0.0, 0.5),
            "reg_alpha": (0.0, 1.0),
            "reg_lambda": (0.0, 1.0),
        }

    def create_model(self, **hp: Any) -> xgb.XGBClassifier | xgb.XGBRegressor:
        """
        Crée une instance XGBoost.

        Args:
            **hp: Hyperparamètres à passer au constructeur

        Returns:
            XGBClassifier ou XGBRegressor selon is_regression
        """
        # Nettoyer les paramètres pour éviter les conflits
        clean_hp = {k: v for k, v in hp.items() if v is not None}

        if self.is_regression:
            return xgb.XGBRegressor(**clean_hp)
        return xgb.XGBClassifier(**clean_hp)
