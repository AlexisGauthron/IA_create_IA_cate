"""
Wrappers sklearn avec interface BaseModel.

Ce module contient les wrappers pour les modèles sklearn classiques :
- RandomForest
- DecisionTree
- LogisticRegression / Ridge
"""

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    """
    Wrapper pour RandomForest avec interface uniforme.

    RandomForest est un ensemble d'arbres de décision, robuste et facile à utiliser.
    Bon baseline pour la plupart des problèmes.

    Example:
        >>> model = RandomForestModel(is_regression=False)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def get_name(self) -> str:
        """Retourne 'randomforest'."""
        return "randomforest"

    def get_default_params(self) -> dict[str, Any]:
        """Hyperparamètres par défaut pour RandomForest."""
        return {
            "n_estimators": 100,
            "max_depth": None,  # Pas de limite par défaut
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": self.random_state,
            "n_jobs": -1,
        }

    def get_hp_space(self) -> dict[str, Any]:
        """Espace de recherche pour HPO."""
        return {
            "n_estimators": (50, 300),
            "max_depth": (5, 30),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 10),
            "max_features": ["sqrt", "log2", 0.5, 0.8],
        }

    def create_model(self, **hp: Any) -> RandomForestClassifier | RandomForestRegressor:
        """Crée une instance RandomForest."""
        clean_hp = {k: v for k, v in hp.items() if v is not None}

        if self.is_regression:
            return RandomForestRegressor(**clean_hp)
        return RandomForestClassifier(**clean_hp)


class DecisionTreeModel(BaseModel):
    """
    Wrapper pour DecisionTree avec interface uniforme.

    DecisionTree est un modèle simple et interprétable. Utile comme baseline
    rapide ou pour le landmarking dans le méta-learning.

    Example:
        >>> model = DecisionTreeModel(is_regression=False)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def get_name(self) -> str:
        """Retourne 'decisiontree'."""
        return "decisiontree"

    def get_default_params(self) -> dict[str, Any]:
        """Hyperparamètres par défaut pour DecisionTree."""
        return {
            "max_depth": 5,  # Limité pour éviter l'overfitting
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": self.random_state,
        }

    def get_hp_space(self) -> dict[str, Any]:
        """Espace de recherche pour HPO."""
        return {
            "max_depth": (3, 20),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 10),
            "criterion": ["gini", "entropy"]
            if not self.is_regression
            else ["squared_error", "friedman_mse"],
        }

    def create_model(self, **hp: Any) -> DecisionTreeClassifier | DecisionTreeRegressor:
        """Crée une instance DecisionTree."""
        clean_hp = {k: v for k, v in hp.items() if v is not None}

        if self.is_regression:
            return DecisionTreeRegressor(**clean_hp)
        return DecisionTreeClassifier(**clean_hp)


class LogisticRegressionModel(BaseModel):
    """
    Wrapper pour LogisticRegression (classification) ou Ridge (régression).

    Modèle linéaire simple et rapide. Excellent baseline et utile pour
    le landmarking dans le méta-learning.

    Note:
        Pour la régression, utilise Ridge au lieu de LinearRegression
        car plus stable avec régularisation.

    Example:
        >>> model = LogisticRegressionModel(is_regression=False)
        >>> model.fit(X_train, y_train)
        >>> probas = model.predict_proba(X_test)
    """

    def get_name(self) -> str:
        """Retourne 'logistic'."""
        return "logistic"

    def get_default_params(self) -> dict[str, Any]:
        """Hyperparamètres par défaut."""
        if self.is_regression:
            return {
                "alpha": 1.0,
                "random_state": self.random_state,
            }
        return {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": self.random_state,
            "n_jobs": -1,
        }

    def get_hp_space(self) -> dict[str, Any]:
        """Espace de recherche pour HPO."""
        if self.is_regression:
            return {
                "alpha": (0.001, 100.0),
            }
        return {
            "C": (0.01, 10.0),
            "penalty": ["l2"],  # l1 nécessite solver='saga'
            "solver": ["lbfgs", "newton-cg"],
        }

    def create_model(self, **hp: Any) -> LogisticRegression | Ridge:
        """Crée une instance LogisticRegression ou Ridge."""
        clean_hp = {k: v for k, v in hp.items() if v is not None}

        if self.is_regression:
            return Ridge(**clean_hp)
        return LogisticRegression(**clean_hp)
