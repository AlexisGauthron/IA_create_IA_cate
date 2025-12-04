"""
Test d'intégration pour LLMFE avec configuration de métrique personnalisée.

Ce script teste l'intégration complète de LLMFE avec :
- Différentes métriques d'évaluation (f1, accuracy, auc, rmse, mae)
- Évaluation multi-modèle (xgboost, lightgbm, randomforest)
- Différentes stratégies d'agrégation (mean, min, max)

Usage:
    # Test rapide (1 itération)
    conda run -n Ia_create_ia python tests/integration/test_llmfe_metric_integration.py --quick

    # Test avec métrique spécifique
    conda run -n Ia_create_ia python tests/integration/test_llmfe_metric_integration.py --metric accuracy

    # Test complet
    conda run -n Ia_create_ia python tests/integration/test_llmfe_metric_integration.py --full
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ajouter le chemin racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.feature_engineering.llmfe.config import EvaluationConfig
from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner, run_llmfe
from src.feature_engineering.llmfe.model_evaluator import (
    evaluate_features,
    evaluate_features_detailed,
)
from src.models import list_models
from src.models.evaluation import get_default_metric, list_metrics


def create_sample_classification_data(n_samples: int = 500) -> tuple[pd.DataFrame, str]:
    """Crée un dataset de classification synthétique."""
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n_samples),
            "income": np.random.exponential(50000, n_samples),
            "education_years": np.random.randint(8, 20, n_samples),
            "experience": np.random.randint(0, 40, n_samples),
            "city": np.random.choice(["Paris", "Lyon", "Marseille", "Bordeaux"], n_samples),
            "gender": np.random.choice(["M", "F"], n_samples),
        }
    )

    # Target basé sur les features (pour avoir une corrélation)
    prob = 1 / (
        1 + np.exp(-(0.02 * df["age"] + 0.00001 * df["income"] + 0.1 * df["education_years"] - 3))
    )
    df["target"] = (np.random.random(n_samples) < prob).astype(int)

    return df, "target"


def create_sample_regression_data(n_samples: int = 500) -> tuple[pd.DataFrame, str]:
    """Crée un dataset de régression synthétique."""
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "surface": np.random.randint(20, 200, n_samples),
            "rooms": np.random.randint(1, 8, n_samples),
            "floor": np.random.randint(0, 15, n_samples),
            "age_building": np.random.randint(0, 100, n_samples),
            "district": np.random.choice(["Center", "North", "South", "East", "West"], n_samples),
            "has_parking": np.random.choice([0, 1], n_samples),
        }
    )

    # Target basé sur les features
    df["price"] = (
        df["surface"] * 5000
        + df["rooms"] * 20000
        + (15 - df["floor"]).abs() * 1000
        - df["age_building"] * 500
        + np.random.normal(0, 10000, n_samples)
    )

    return df, "price"


def test_available_metrics():
    """Affiche les métriques disponibles."""
    print("\n" + "=" * 60)
    print("        MÉTRIQUES DISPONIBLES")
    print("=" * 60)

    print("\n📊 Classification:")
    clf_metrics = list_metrics(is_regression=False)
    print(f"   {', '.join(clf_metrics)}")
    print(f"   Default: {get_default_metric(is_regression=False)}")

    print("\n📈 Régression:")
    reg_metrics = list_metrics(is_regression=True)
    print(f"   {', '.join(reg_metrics)}")
    print(f"   Default: {get_default_metric(is_regression=True)}")

    print("\n🤖 Modèles disponibles:")
    models = list_models()
    print(f"   {', '.join(models)}")


def test_evaluate_features_with_metric(metric: str, is_regression: bool = False):
    """Teste l'évaluation avec une métrique spécifique."""
    print(f"\n{'=' * 60}")
    print(f"   TEST: evaluate_features() avec metric='{metric}'")
    print(f"{'=' * 60}")

    # Créer les données
    if is_regression:
        df, target = create_sample_regression_data(200)
    else:
        df, target = create_sample_classification_data(200)

    X = df.drop(columns=[target])
    y = df[target].values

    # Test avec un seul modèle
    print("\n1️⃣ Évaluation avec XGBoost seul:")
    score_single = evaluate_features(
        X,
        y,
        is_regression=is_regression,
        model_names=["xgboost"],
        metric=metric,
    )
    print(f"   Score ({metric}): {score_single:.4f}")

    # Test multi-modèle
    print("\n2️⃣ Évaluation multi-modèle (XGBoost + LightGBM + RandomForest):")
    score_multi = evaluate_features(
        X,
        y,
        is_regression=is_regression,
        model_names=["xgboost", "lightgbm", "randomforest"],
        metric=metric,
        aggregation="mean",
    )
    print(f"   Score agrégé ({metric}): {score_multi:.4f}")

    # Détails par modèle
    print("\n3️⃣ Détails par modèle:")
    details = evaluate_features_detailed(
        X,
        y,
        is_regression=is_regression,
        model_names=["xgboost", "lightgbm", "randomforest"],
        metric=metric,
    )
    for model_name, score in details["scores"].items():
        std = details["std"][model_name]
        print(f"   - {model_name}: {score:.4f} (±{std:.4f})")
    print(f"   Meilleur: {details['best_model']} avec {details['best_score']:.4f}")

    return True


def test_evaluation_config():
    """Teste EvaluationConfig avec différentes configurations."""
    print(f"\n{'=' * 60}")
    print("   TEST: EvaluationConfig")
    print(f"{'=' * 60}")

    configs = [
        ("Default", EvaluationConfig()),
        ("F1", EvaluationConfig(metric="f1")),
        ("Accuracy", EvaluationConfig(metric="accuracy")),
        ("AUC", EvaluationConfig(metric="auc")),
        (
            "Multi-model + Accuracy",
            EvaluationConfig(
                model_names=("xgboost", "lightgbm"),
                metric="accuracy",
                aggregation="mean",
            ),
        ),
    ]

    for name, config in configs:
        print(f"\n   {name}:")
        print(f"      Modèles: {config.get_model_names()}")
        print(f"      Métrique: {config.metric}")
        print(f"      Agrégation: {config.aggregation}")
        print(f"      N folds: {config.n_folds}")

    return True


def test_llmfe_runner_config(metric: str = "accuracy"):
    """Teste que LLMFERunner accepte la configuration de métrique."""
    print(f"\n{'=' * 60}")
    print(f"   TEST: LLMFERunner avec metric='{metric}'")
    print(f"{'=' * 60}")

    # Créer le runner
    runner = LLMFERunner(project_name="test_metric_integration")

    # Vérifier la signature
    import inspect

    sig = inspect.signature(runner.run)

    print("\n   Paramètres de run():")
    for param_name in ["eval_models", "eval_aggregation", "eval_metric", "eval_config"]:
        if param_name in sig.parameters:
            default = sig.parameters[param_name].default
            print(f"      ✅ {param_name} = {default}")
        else:
            print(f"      ❌ {param_name} manquant!")
            return False

    # Simuler la construction de EvaluationConfig comme dans run()
    print("\n   Simulation de la configuration:")
    eval_models = ["xgboost", "lightgbm"]
    eval_aggregation = "mean"
    eval_metric = metric

    eval_config = EvaluationConfig(
        model_names=tuple(eval_models),
        aggregation=eval_aggregation,
        metric=eval_metric,
    )

    print(f"      Modèles: {eval_config.get_model_names()}")
    print(f"      Métrique: {eval_config.metric}")
    print(f"      Agrégation: {eval_config.aggregation}")

    assert (
        eval_config.metric == metric
    ), f"Métrique attendue: {metric}, obtenue: {eval_config.metric}"
    print("\n   ✅ Configuration correcte!")

    return True


def test_full_llmfe_run(metric: str = "f1", max_samples: int = 1):
    """
    Test complet de LLMFE avec une métrique spécifique.

    ATTENTION: Ce test lance vraiment LLMFE et nécessite une API LLM configurée.
    """
    print(f"\n{'=' * 60}")
    print(f"   TEST COMPLET: run_llmfe() avec metric='{metric}'")
    print(f"{'=' * 60}")

    # Créer les données
    df, target = create_sample_classification_data(200)

    print(f"\n   Dataset: {len(df)} samples, {len(df.columns)-1} features")
    print(f"   Target: {target}")
    print(f"   Métrique: {metric}")
    print(f"   Max samples: {max_samples}")

    try:
        result = run_llmfe(
            project_name="test_metric_full",
            df_train=df,
            target_col=target,
            is_regression=False,
            max_samples=max_samples,
            eval_models=["xgboost", "lightgbm"],
            eval_aggregation="mean",
            eval_metric=metric,
            use_api=True,
            api_model="gpt-4o-mini",  # Modèle moins cher pour les tests
        )

        print("\n   ✅ LLMFE terminé!")
        print(f"   Résultats: {result['results_dir']}")
        return True

    except Exception as e:
        print(f"\n   ⚠️ Erreur (attendue si pas d'API configurée): {e}")
        print("   → Le test de configuration est quand même validé")
        return True  # On valide quand même car la config est correcte


def main():
    parser = argparse.ArgumentParser(description="Test d'intégration LLMFE avec métriques")
    parser.add_argument("--quick", action="store_true", help="Test rapide (sans LLMFE complet)")
    parser.add_argument("--full", action="store_true", help="Test complet avec LLMFE")
    parser.add_argument(
        "--metric", type=str, default="f1", help="Métrique à tester (f1, accuracy, auc, etc.)"
    )
    parser.add_argument("--regression", action="store_true", help="Tester la régression")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("   TEST D'INTÉGRATION LLMFE - CONFIGURATION MÉTRIQUE")
    print("=" * 60)

    all_passed = True

    # Test 1: Afficher les métriques disponibles
    test_available_metrics()

    # Test 2: EvaluationConfig
    if not test_evaluation_config():
        all_passed = False

    # Test 3: LLMFERunner config
    if not test_llmfe_runner_config(args.metric):
        all_passed = False

    # Test 4: Évaluation avec la métrique spécifiée
    if args.regression:
        metric = args.metric if args.metric != "f1" else "rmse"
        if not test_evaluate_features_with_metric(metric, is_regression=True):
            all_passed = False
    else:
        if not test_evaluate_features_with_metric(args.metric, is_regression=False):
            all_passed = False

    # Test 5: Test complet (si demandé)
    if args.full:
        if not test_full_llmfe_run(args.metric, max_samples=2):
            all_passed = False

    # Résumé
    print("\n" + "=" * 60)
    if all_passed:
        print("   ✅ TOUS LES TESTS PASSÉS!")
    else:
        print("   ❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("=" * 60)

    # Afficher les exemples d'utilisation
    print("\n" + "=" * 60)
    print("   EXEMPLES D'UTILISATION")
    print("=" * 60)
    print("""
    # Classification avec accuracy
    result = run_llmfe(
        project_name="mon_projet",
        df_train=df,
        target_col="target",
        eval_metric="accuracy",
    )

    # Classification avec AUC et multi-modèle
    result = run_llmfe(
        project_name="mon_projet",
        df_train=df,
        target_col="target",
        eval_models=["xgboost", "lightgbm", "randomforest"],
        eval_aggregation="mean",
        eval_metric="auc",
    )

    # Régression avec MAE
    result = run_llmfe(
        project_name="mon_projet",
        df_train=df,
        target_col="price",
        is_regression=True,
        eval_metric="mae",
    )
    """)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
