"""
Test unitaire pour la configuration de la métrique d'évaluation dans LLMFE.

Ce test vérifie que le paramètre eval_metric est correctement propagé
à travers LLMFERunner et EvaluationConfig.

Usage:
    conda run -n Ia_create_ia pytest tests/unit/test_eval_metric_config.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.llmfe.config import EvaluationConfig
from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner
from src.models.evaluation import get_default_metric, list_metrics


class TestEvaluationConfigMetric:
    """Tests pour EvaluationConfig avec différentes métriques."""

    def test_default_metric_is_auto(self):
        """La métrique par défaut doit être 'auto'."""
        config = EvaluationConfig()
        assert config.metric == "auto"

    def test_custom_metric_f1(self):
        """Test avec métrique f1."""
        config = EvaluationConfig(metric="f1")
        assert config.metric == "f1"

    def test_custom_metric_accuracy(self):
        """Test avec métrique accuracy."""
        config = EvaluationConfig(metric="accuracy")
        assert config.metric == "accuracy"

    def test_custom_metric_auc(self):
        """Test avec métrique auc."""
        config = EvaluationConfig(metric="auc")
        assert config.metric == "auc"

    def test_custom_metric_rmse(self):
        """Test avec métrique rmse (régression)."""
        config = EvaluationConfig(metric="rmse")
        assert config.metric == "rmse"

    def test_full_config_with_metric(self):
        """Test configuration complète avec tous les paramètres."""
        config = EvaluationConfig(
            model_names=("xgboost", "lightgbm"),
            n_folds=5,
            metric="accuracy",
            aggregation="mean",
        )
        assert config.metric == "accuracy"
        assert config.n_folds == 5
        assert config.get_model_names() == ["xgboost", "lightgbm"]
        assert config.aggregation == "mean"


class TestLLMFERunnerMetric:
    """Tests pour LLMFERunner avec le paramètre eval_metric."""

    @pytest.fixture
    def sample_df(self):
        """Crée un DataFrame de test."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.choice(["A", "B", "C"], 100),
                "target": np.random.choice([0, 1], 100),
            }
        )

    def test_runner_accepts_eval_metric(self, sample_df):
        """Vérifie que LLMFERunner accepte eval_metric."""
        runner = LLMFERunner(project_name="test_metric")

        # Vérifier que la signature de run() accepte eval_metric
        import inspect

        sig = inspect.signature(runner.run)
        assert "eval_metric" in sig.parameters
        assert sig.parameters["eval_metric"].default == "auto"

    def test_eval_config_receives_metric(self):
        """Vérifie que EvaluationConfig reçoit bien la métrique."""
        # Simuler la construction comme dans run()
        eval_metric = "accuracy"
        eval_models = ["xgboost", "lightgbm"]
        eval_aggregation = "mean"

        eval_config = EvaluationConfig(
            model_names=tuple(eval_models),
            aggregation=eval_aggregation,
            metric=eval_metric,
        )

        assert eval_config.metric == "accuracy"
        assert eval_config.get_model_names() == ["xgboost", "lightgbm"]

    def test_eval_config_default_metric(self):
        """Vérifie la métrique par défaut dans EvaluationConfig."""
        eval_metric = "auto"

        eval_config = EvaluationConfig(metric=eval_metric)

        assert eval_config.metric == "auto"


class TestAvailableMetrics:
    """Tests pour vérifier les métriques disponibles."""

    def test_classification_metrics_available(self):
        """Vérifie que les métriques de classification sont disponibles."""
        metrics = list_metrics(is_regression=False)
        expected = [
            "accuracy",
            "f1",
            "f1_macro",
            "f1_micro",
            "precision",
            "recall",
            "auc",
            "logloss",
        ]
        for metric in expected:
            assert metric in metrics, f"Métrique {metric} non disponible"

    def test_regression_metrics_available(self):
        """Vérifie que les métriques de régression sont disponibles."""
        metrics = list_metrics(is_regression=True)
        expected = ["rmse", "mse", "mae", "r2", "mape"]
        for metric in expected:
            assert metric in metrics, f"Métrique {metric} non disponible"

    def test_default_metric_classification(self):
        """Vérifie la métrique par défaut pour la classification."""
        default = get_default_metric(is_regression=False)
        assert default == "f1"

    def test_default_metric_regression(self):
        """Vérifie la métrique par défaut pour la régression."""
        default = get_default_metric(is_regression=True)
        assert default == "rmse"


class TestRunLLMFEFunction:
    """Tests pour la fonction run_llmfe()."""

    def test_run_llmfe_accepts_eval_metric(self):
        """Vérifie que run_llmfe() accepte eval_metric."""
        import inspect

        from src.feature_engineering.llmfe.llmfe_runner import run_llmfe

        sig = inspect.signature(run_llmfe)
        assert "eval_metric" in sig.parameters
        assert sig.parameters["eval_metric"].default == "auto"


class TestMetricIntegration:
    """Tests d'intégration pour la métrique dans l'évaluation."""

    @pytest.fixture
    def classification_data(self):
        """Données de classification pour les tests."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "f1": np.random.randn(100),
                "f2": np.random.randn(100),
            }
        )
        y = np.random.choice([0, 1], 100)
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Données de régression pour les tests."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "f1": np.random.randn(100),
                "f2": np.random.randn(100),
            }
        )
        y = np.random.randn(100)
        return X, y

    def test_evaluate_with_f1(self, classification_data):
        """Test d'évaluation avec métrique f1."""
        from src.feature_engineering.llmfe.model_evaluator import evaluate_features

        X, y = classification_data
        score = evaluate_features(X, y, is_regression=False, metric="f1")
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_evaluate_with_accuracy(self, classification_data):
        """Test d'évaluation avec métrique accuracy."""
        from src.feature_engineering.llmfe.model_evaluator import evaluate_features

        X, y = classification_data
        score = evaluate_features(X, y, is_regression=False, metric="accuracy")
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_evaluate_with_rmse(self, regression_data):
        """Test d'évaluation avec métrique rmse."""
        from src.feature_engineering.llmfe.model_evaluator import evaluate_features

        X, y = regression_data
        score = evaluate_features(X, y, is_regression=True, metric="rmse")
        assert isinstance(score, float)
        # RMSE est toujours positif (erreur quadratique moyenne)
        assert score >= 0

    def test_evaluate_with_mae(self, regression_data):
        """Test d'évaluation avec métrique mae."""
        from src.feature_engineering.llmfe.model_evaluator import evaluate_features

        X, y = regression_data
        score = evaluate_features(X, y, is_regression=True, metric="mae")
        assert isinstance(score, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
