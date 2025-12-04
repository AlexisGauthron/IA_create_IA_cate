"""
Tests unitaires pour le module src/models/ et l'intégration avec LLMFE.

Ce fichier teste :
1. Le module src/models/ (BaseModel, registry, CrossValidator)
2. L'évaluateur multi-modèle pour LLMFE
3. La configuration d'évaluation

Usage:
    # Lancer tous les tests
    conda run -n Ia_create_ia pytest tests/unit/test_models_module.py -v

    # Lancer un test spécifique
    conda run -n Ia_create_ia pytest tests/unit/test_models_module.py::test_evaluate_features_multi_model -v

    # Lancer avec affichage détaillé
    conda run -n Ia_create_ia pytest tests/unit/test_models_module.py -v -s
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

# ============================================================================
# TESTS DU MODULE src/models/
# ============================================================================


class TestModelsRegistry:
    """Tests pour le registre de modèles."""

    def test_list_models(self):
        """Vérifie que les modèles sont listés correctement."""
        from src.models import list_models

        models = list_models()
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "randomforest" in models
        assert "decisiontree" in models
        assert "logistic" in models
        print(f"Modèles disponibles: {models}")

    def test_get_model_classification(self):
        """Vérifie qu'on peut récupérer un modèle de classification."""
        from src.models import get_model

        model = get_model("xgboost", is_regression=False)
        assert model.get_name() == "xgboost"
        assert model.is_regression is False
        print(f"Modèle créé: {model.get_name()}")

    def test_get_model_regression(self):
        """Vérifie qu'on peut récupérer un modèle de régression."""
        from src.models import get_model

        model = get_model("lightgbm", is_regression=True)
        assert model.get_name() == "lightgbm"
        assert model.is_regression is True

    def test_get_models_multiple(self):
        """Vérifie qu'on peut récupérer plusieurs modèles."""
        from src.models import get_models

        models = get_models(["xgboost", "lightgbm", "randomforest"])
        assert len(models) == 3
        names = [m.get_name() for m in models]
        assert "xgboost" in names
        assert "lightgbm" in names
        assert "randomforest" in names

    def test_get_all_models(self):
        """Vérifie qu'on peut récupérer tous les modèles."""
        from src.models import get_all_models, list_models

        all_models = get_all_models()
        assert len(all_models) == len(list_models())

    def test_unknown_model_raises_error(self):
        """Vérifie qu'un modèle inconnu lève une erreur."""
        from src.models import get_model

        with pytest.raises(ValueError, match="Modèle inconnu"):
            get_model("unknown_model")


class TestBaseModel:
    """Tests pour l'interface BaseModel."""

    @pytest.fixture
    def sample_data_classification(self):
        """Données de classification pour les tests."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)]), y

    @pytest.fixture
    def sample_data_regression(self):
        """Données de régression pour les tests."""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)]), y

    def test_model_fit_predict(self, sample_data_classification):
        """Vérifie que fit/predict fonctionnent."""
        from src.models import get_model

        X, y = sample_data_classification
        model = get_model("xgboost", is_regression=False)

        # Fit
        model.fit(X, y)
        assert model.model is not None

        # Predict
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_model_predict_proba(self, sample_data_classification):
        """Vérifie que predict_proba fonctionne."""
        from src.models import get_model

        X, y = sample_data_classification
        model = get_model("randomforest", is_regression=False)
        model.fit(X, y)

        probas = model.predict_proba(X)
        assert probas.shape == (len(y), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_model_clone(self, sample_data_classification):
        """Vérifie que clone fonctionne."""
        from src.models import get_model

        X, y = sample_data_classification
        model = get_model("lightgbm", is_regression=False)
        model.fit(X, y)

        # Clone
        cloned = model.clone()
        assert cloned.model is None  # Non entraîné
        assert cloned.get_name() == model.get_name()
        assert cloned.is_regression == model.is_regression

    def test_model_default_params(self):
        """Vérifie que les paramètres par défaut sont définis."""
        from src.models import get_model

        model = get_model("xgboost")
        params = model.get_default_params()
        assert isinstance(params, dict)
        assert "n_estimators" in params or "iterations" in params
        print(f"Paramètres par défaut XGBoost: {params}")

    def test_model_hp_space(self):
        """Vérifie que l'espace HP est défini."""
        from src.models import get_model

        model = get_model("lightgbm")
        hp_space = model.get_hp_space()
        assert isinstance(hp_space, dict)
        assert len(hp_space) > 0
        print(f"Espace HP LightGBM: {list(hp_space.keys())}")


class TestCrossValidator:
    """Tests pour le CrossValidator."""

    @pytest.fixture
    def sample_data(self):
        """Données pour les tests."""
        X, y = make_classification(n_samples=300, n_features=10, n_informative=5, random_state=42)
        return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)]), y

    def test_evaluate_single_model(self, sample_data):
        """Vérifie l'évaluation d'un seul modèle."""
        from src.models import CrossValidator, get_model

        X, y = sample_data
        cv = CrossValidator(n_folds=3, random_state=42)
        model = get_model("xgboost", is_regression=False)

        result = cv.evaluate(model, X, y, metric="f1")

        assert 0 <= result.mean <= 1
        assert result.std >= 0
        assert len(result.scores) == 3
        assert result.model_name == "xgboost"
        print(f"Score XGBoost: {result.mean:.4f} (+/- {result.std:.4f})")

    def test_evaluate_multi_model(self, sample_data):
        """Vérifie l'évaluation multi-modèle."""
        from src.models import CrossValidator, get_models

        X, y = sample_data
        cv = CrossValidator(n_folds=3, random_state=42)
        models = get_models(["xgboost", "lightgbm", "randomforest"])

        result = cv.evaluate_multi_model(models, X, y, aggregation="mean")

        assert 0 <= result.aggregated_score <= 1
        assert result.best_model in ["xgboost", "lightgbm", "randomforest"]
        assert len(result.results) == 3

        print(f"Score agrégé: {result.aggregated_score:.4f}")
        print(f"Meilleur modèle: {result.best_model}")
        for name, res in result.results.items():
            print(f"  {name}: {res.mean:.4f}")

    def test_quick_evaluate(self, sample_data):
        """Vérifie l'évaluation rapide."""
        from src.models import CrossValidator, get_model

        X, y = sample_data
        cv = CrossValidator(n_folds=3, random_state=42)
        model = get_model("decisiontree", is_regression=False)

        score = cv.quick_evaluate(model, X, y, sample_size=100)
        assert 0 <= score <= 1
        print(f"Score rapide: {score:.4f}")


class TestMetrics:
    """Tests pour les métriques."""

    def test_list_metrics(self):
        """Vérifie la liste des métriques."""
        from src.models import list_metrics

        clf_metrics = list_metrics(is_regression=False)
        reg_metrics = list_metrics(is_regression=True)

        assert "f1" in clf_metrics
        assert "accuracy" in clf_metrics
        assert "auc" in clf_metrics

        assert "rmse" in reg_metrics
        assert "r2" in reg_metrics

        print(f"Métriques classification: {clf_metrics}")
        print(f"Métriques régression: {reg_metrics}")

    def test_get_metric(self):
        """Vérifie la récupération d'une métrique."""
        from src.models import get_metric

        f1_fn = get_metric("f1", is_regression=False)
        assert callable(f1_fn)

        rmse_fn = get_metric("rmse", is_regression=True)
        assert callable(rmse_fn)


# ============================================================================
# TESTS DE L'INTÉGRATION LLMFE
# ============================================================================


class TestLLMFEModelEvaluator:
    """Tests pour l'évaluateur multi-modèle LLMFE."""

    @pytest.fixture
    def sample_data(self):
        """Données pour les tests."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)]), y

    def test_evaluate_features_single_model(self, sample_data):
        """Vérifie l'évaluation avec un seul modèle (legacy)."""
        from src.feature_engineering.llmfe.model_evaluator import evaluate_features

        X, y = sample_data
        score = evaluate_features(X, y, is_regression=False, model_names=["xgboost"])

        assert 0 <= score <= 1
        print(f"Score XGBoost seul: {score:.4f}")

    def test_evaluate_features_multi_model(self, sample_data):
        """Vérifie l'évaluation multi-modèle."""
        from src.feature_engineering.llmfe.model_evaluator import evaluate_features

        X, y = sample_data
        score = evaluate_features(
            X,
            y,
            is_regression=False,
            model_names=["xgboost", "lightgbm", "randomforest"],
            aggregation="mean",
        )

        assert 0 <= score <= 1
        print(f"Score multi-modèle (mean): {score:.4f}")

    def test_evaluate_features_detailed(self, sample_data):
        """Vérifie l'évaluation détaillée."""
        from src.feature_engineering.llmfe.model_evaluator import (
            evaluate_features_detailed,
        )

        X, y = sample_data
        result = evaluate_features_detailed(
            X, y, model_names=["xgboost", "lightgbm", "randomforest"]
        )

        assert "scores" in result
        assert "best_model" in result
        assert "best_score" in result
        assert len(result["scores"]) == 3

        print("Résultats détaillés:")
        for name, score in result["scores"].items():
            print(f"  {name}: {score:.4f}")
        print(f"Meilleur: {result['best_model']} ({result['best_score']:.4f})")

    def test_evaluate_features_with_categorical(self):
        """Vérifie que les colonnes catégorielles sont gérées."""
        from src.feature_engineering.llmfe.model_evaluator import evaluate_features

        # Créer des données avec colonnes catégorielles
        X = pd.DataFrame(
            {
                "num1": np.random.randn(100),
                "num2": np.random.randn(100),
                "cat1": np.random.choice(["A", "B", "C"], 100),
                "cat2": np.random.choice(["X", "Y"], 100),
            }
        )
        y = np.random.randint(0, 2, 100)

        score = evaluate_features(X, y, is_regression=False, model_names=["xgboost"])
        assert 0 <= score <= 1
        print(f"Score avec catégorielles: {score:.4f}")

    def test_evaluate_features_with_missing(self):
        """Vérifie que les valeurs manquantes sont gérées."""
        from src.feature_engineering.llmfe.model_evaluator import evaluate_features

        # Créer des données avec valeurs manquantes
        X = pd.DataFrame(
            {
                "feat1": [1, 2, np.nan, 4, 5] * 20,
                "feat2": [np.nan, 2, 3, 4, 5] * 20,
                "feat3": [1, 2, 3, np.inf, 5] * 20,
            }
        )
        y = np.random.randint(0, 2, 100)

        score = evaluate_features(X, y, is_regression=False, model_names=["xgboost"])
        assert 0 <= score <= 1
        print(f"Score avec valeurs manquantes: {score:.4f}")


class TestLLMFEConfig:
    """Tests pour la configuration LLMFE."""

    def test_evaluation_config_default(self):
        """Vérifie la configuration par défaut."""
        from src.feature_engineering.llmfe.config import EvaluationConfig

        config = EvaluationConfig()
        assert config.get_model_names() == ["xgboost"]
        assert config.n_folds == 4
        assert config.aggregation == "mean"

    def test_evaluation_config_custom(self):
        """Vérifie une configuration personnalisée."""
        from src.feature_engineering.llmfe.config import EvaluationConfig

        config = EvaluationConfig(
            model_names=("xgboost", "lightgbm"),
            n_folds=5,
            metric="f1",
            aggregation="min",
        )

        assert config.get_model_names() == ["xgboost", "lightgbm"]
        assert config.n_folds == 5
        assert config.metric == "f1"
        assert config.aggregation == "min"

    def test_evaluation_presets(self):
        """Vérifie les presets de configuration."""
        from src.feature_engineering.llmfe.config import (
            EVAL_FAST,
            EVAL_LEGACY,
            EVAL_MULTI_MODEL,
        )

        assert EVAL_LEGACY.get_model_names() == ["xgboost"]
        assert "xgboost" in EVAL_MULTI_MODEL.get_model_names()
        assert "lightgbm" in EVAL_MULTI_MODEL.get_model_names()
        assert "decisiontree" in EVAL_FAST.get_model_names()

        print(f"EVAL_LEGACY: {EVAL_LEGACY.get_model_names()}")
        print(f"EVAL_MULTI_MODEL: {EVAL_MULTI_MODEL.get_model_names()}")
        print(f"EVAL_FAST: {EVAL_FAST.get_model_names()}")


class TestLLMFERunner:
    """Tests pour le runner LLMFE."""

    def test_runner_initialization(self):
        """Vérifie l'initialisation du runner."""
        from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner

        runner = LLMFERunner(project_name="test_project")
        assert runner.project_name == "test_project"

    def test_generate_spec_legacy(self):
        """Vérifie la génération de spec legacy."""
        from src.feature_engineering.llmfe.config import EvaluationConfig
        from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner

        runner = LLMFERunner(project_name="test")
        config = EvaluationConfig()  # Legacy

        spec = runner._generate_spec(
            task_description="Test task",
            is_regression=False,
            eval_config=config,
        )

        assert "evaluate_features" in spec
        assert "model_names=['xgboost']" in spec
        print("Spec legacy générée avec succès")

    def test_generate_spec_multi_model(self):
        """Vérifie la génération de spec multi-modèle."""
        from src.feature_engineering.llmfe.config import EvaluationConfig
        from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner

        runner = LLMFERunner(project_name="test")
        config = EvaluationConfig(
            model_names=("xgboost", "lightgbm", "randomforest"),
            aggregation="mean",
        )

        spec = runner._generate_spec(
            task_description="Test task",
            is_regression=False,
            eval_config=config,
        )

        assert "evaluate_features" in spec
        assert "xgboost" in spec
        assert "lightgbm" in spec
        assert "randomforest" in spec
        assert 'aggregation="mean"' in spec
        print("Spec multi-modèle générée avec succès")


# ============================================================================
# TEST D'INTÉGRATION COMPLET
# ============================================================================


class TestFullIntegration:
    """Test d'intégration complet de bout en bout."""

    def test_full_evaluation_pipeline(self):
        """
        Test complet du pipeline d'évaluation.

        Simule ce qui se passe dans LLMFE quand on évalue des features.
        """
        from src.feature_engineering.llmfe.config import EvaluationConfig
        from src.feature_engineering.llmfe.model_evaluator import (
            evaluate_features,
            evaluate_features_detailed,
        )

        print("\n" + "=" * 60)
        print("TEST D'INTÉGRATION COMPLET")
        print("=" * 60)

        # 1. Créer des données de test
        print("\n1. Création des données de test...")
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=10,
            n_classes=2,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
        print(f"   Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")

        # 2. Configuration legacy (XGBoost seul)
        print("\n2. Évaluation legacy (XGBoost seul)...")
        score_legacy = evaluate_features(X_df, y, is_regression=False, model_names=["xgboost"])
        print(f"   Score: {score_legacy:.4f}")

        # 3. Configuration multi-modèle
        print("\n3. Évaluation multi-modèle...")
        config = EvaluationConfig(
            model_names=("xgboost", "lightgbm", "randomforest"),
            n_folds=4,
            aggregation="mean",
        )

        result = evaluate_features_detailed(
            X_df,
            y,
            is_regression=False,
            model_names=config.get_model_names(),
            n_folds=config.n_folds,
        )

        print("   Scores par modèle:")
        for name, score in result["scores"].items():
            std = result["std"][name]
            print(f"     - {name}: {score:.4f} (+/- {std:.4f})")

        print(f"\n   Meilleur modèle: {result['best_model']}")
        print(f"   Meilleur score: {result['best_score']:.4f}")

        # 4. Comparaison des agrégations
        print("\n4. Comparaison des stratégies d'agrégation...")
        for agg in ["mean", "min", "max"]:
            score = evaluate_features(
                X_df,
                y,
                model_names=["xgboost", "lightgbm", "randomforest"],
                aggregation=agg,
            )
            print(f"   Agrégation '{agg}': {score:.4f}")

        print("\n" + "=" * 60)
        print("TEST D'INTÉGRATION RÉUSSI !")
        print("=" * 60)

        # Assertions
        assert 0.7 <= score_legacy <= 1.0, "Score legacy trop bas"
        assert result["best_model"] in ["xgboost", "lightgbm", "randomforest"]


# ============================================================================
# POINT D'ENTRÉE POUR EXÉCUTION DIRECTE
# ============================================================================

if __name__ == "__main__":
    """
    Permet d'exécuter les tests directement avec Python.

    Usage:
        conda run -n Ia_create_ia python tests/unit/test_models_module.py
    """
    import sys

    # Exécuter pytest programmatiquement
    sys.exit(pytest.main([__file__, "-v", "-s"]))
