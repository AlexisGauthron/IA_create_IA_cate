"""
Wrapper CatBoost avec interface BaseModel.
"""

from typing import Any

from src.models.base import BaseModel

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None  # type: ignore
    CatBoostRegressor = None  # type: ignore


class CatBoostModel(BaseModel):
    """
    Wrapper pour CatBoost avec interface uniforme.

    CatBoost est un algorithme de gradient boosting développé par Yandex,
    particulièrement bon pour les données avec beaucoup de features catégorielles.
    Il gère nativement les catégories sans encodage manuel.

    Note:
        CatBoost est optionnel. Si non installé, une erreur sera levée.

    Example:
        >>> model = CatBoostModel(is_regression=False)
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
        Initialise le modèle CatBoost.

        Raises:
            ImportError: Si CatBoost n'est pas installé
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost n'est pas installé. "
                "Installez-le avec: pip install catboost ou conda install -c conda-forge catboost"
            )
        super().__init__(is_regression=is_regression, random_state=random_state, **kwargs)

    def get_name(self) -> str:
        """Retourne 'catboost'."""
        return "catboost"

    def get_default_params(self) -> dict[str, Any]:
        """
        Hyperparamètres par défaut pour CatBoost.

        Returns:
            Dictionnaire des paramètres optimisés.
        """
        return {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
            "random_seed": self.random_state,
            "verbose": False,
            "allow_writing_files": False,  # Évite la création de fichiers temporaires
            "thread_count": -1,
        }

    def get_hp_space(self) -> dict[str, Any]:
        """
        Espace de recherche des hyperparamètres pour HPO.

        Returns:
            Dictionnaire définissant les ranges pour Optuna.
        """
        return {
            "iterations": (50, 500),
            "depth": (4, 10),
            "learning_rate": (0.01, 0.3),
            "l2_leaf_reg": (1.0, 10.0),
            "bagging_temperature": (0.0, 1.0),
            "random_strength": (0.0, 10.0),
        }

    def create_model(self, **hp: Any) -> Any:
        """
        Crée une instance CatBoost.

        Args:
            **hp: Hyperparamètres à passer au constructeur

        Returns:
            CatBoostClassifier ou CatBoostRegressor selon is_regression
        """
        # Nettoyer les paramètres
        clean_hp = {k: v for k, v in hp.items() if v is not None}

        if self.is_regression:
            return CatBoostRegressor(**clean_hp)
        return CatBoostClassifier(**clean_hp)
