#!/usr/bin/env python
"""
Test d'intégration pour le pipeline complet.

Pour la documentation complète des paramètres CLI, voir: docs/cli_reference.md

Usage rapide:
    # Analyse seule (REQUIS: --dataset et --target)
    python tests/integration/test_pipeline_all.py --dataset titanic --target Survived --analyse-only

    # Pipeline complet
    python tests/integration/test_pipeline_all.py --dataset titanic --target Survived --full

    # Avec overrides
    python tests/integration/test_pipeline_all.py --dataset titanic --target Survived --full \
        --override-metric f1 --override-time-budget 60
"""

import argparse
import os
import sys
from pathlib import Path

# Ajouter le dossier racine au path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# =============================================================================
# IMPORTANT: Charger le .env AVANT tout import de modules utilisant des API keys
# =============================================================================
from dotenv import load_dotenv

env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"[CONFIG] Fichier .env chargé depuis {env_path}")

    # LLMFE utilise API_KEY comme nom de variable (pas OPENAI_API_KEY)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        os.environ["API_KEY"] = openai_key
        print("[CONFIG] API_KEY configurée pour LLMFE")
    else:
        print("[CONFIG] ⚠️ OPENAI_API_KEY non trouvée dans .env")
else:
    print(f"[CONFIG] ⚠️ Fichier .env non trouvé à {env_path}")

import pandas as pd

# =============================================================================
# Fonctions utilitaires
# =============================================================================


def load_data(dataset_name: str) -> pd.DataFrame:
    """
    Charge un dataset depuis data/raw/{dataset_name}/.

    Cherche automatiquement train.csv dans le dossier correspondant.
    Si le dataset n'existe pas, liste les datasets disponibles.
    """
    data_raw_dir = ROOT_DIR / "data" / "raw"
    dataset_dir = data_raw_dir / dataset_name

    # Vérifier si le dossier du dataset existe
    if not dataset_dir.exists():
        available_datasets = [
            d.name for d in data_raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' non trouvé dans data/raw/.\n"
            f"Datasets disponibles: {available_datasets}"
        )

    # Chercher train.csv dans le dossier
    data_path = dataset_dir / "train.csv"

    if not data_path.exists():
        available_files = [f.name for f in dataset_dir.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"Fichier 'train.csv' non trouvé dans data/raw/{dataset_name}/.\n"
            f"Fichiers disponibles: {available_files}"
        )

    # Détecter le séparateur automatiquement
    with open(data_path, encoding="utf-8") as f:
        first_line = f.readline()

    # Compter les séparateurs potentiels
    separators = {
        ",": first_line.count(","),
        ";": first_line.count(";"),
        "\t": first_line.count("\t"),
    }
    sep = max(separators, key=separators.get) if max(separators.values()) > 0 else ","

    df = pd.read_csv(data_path, sep=sep)
    print(
        f"Dataset '{dataset_name}' chargé: {len(df)} lignes, {len(df.columns)} colonnes (sep='{sep}')"
    )
    return df


def parse_list(value: str) -> list:
    """Parse une liste séparée par des virgules."""
    if not value:
        return []
    return [v.strip() for v in value.split(",")]


# =============================================================================
# Fonctions de test
# =============================================================================


def _filter_kwargs(kwargs: dict) -> dict:
    """Filtre les kwargs internes (préfixés par _) qui ne sont pas acceptés par run_pipeline."""
    return {k: v for k, v in kwargs.items() if not k.startswith("_")}


def test_pipeline_analyse_only(projet_name: str, target_col: str, df: pd.DataFrame, **kwargs):
    """Test de l'analyse seule (sans FE ni AutoML)."""
    print("\n" + "=" * 70)
    print("TEST: Analyse seule")
    print("=" * 70)

    from src.pipeline.pipeline_all import run_pipeline

    # Filtrer les kwargs internes (préfixés par _)
    filtered_kwargs = _filter_kwargs(kwargs)

    result = run_pipeline(
        project_name=projet_name,
        df_train=df,
        target_col=target_col,
        enable_fe=False,
        enable_automl=False,
        **filtered_kwargs,
    )

    # Vérifications
    assert result.analyse_result is not None, "Analyse devrait être complétée"
    assert result.detected_params is not None, "Params devraient être détectés"

    print("\n--- Résultats ---")
    print(f"Task type détecté: {result.detected_params.task_type}")
    print(f"Metric détectée: {result.detected_params.metric}")
    print(f"Feature format: {result.detected_params.feature_format}")
    print(f"Problem type: {result.detected_params.problem_type}")
    print(f"Is imbalanced: {result.detected_params.is_imbalanced}")

    return result


def test_pipeline_analyse_and_fe(projet_name: str, target_col: str, df: pd.DataFrame, **kwargs):
    """Test de l'analyse + Feature Engineering (sans AutoML)."""
    print("\n" + "=" * 70)
    print("TEST: Analyse + Feature Engineering")
    print("=" * 70)

    from src.pipeline.pipeline_all import run_pipeline

    # Filtrer les kwargs internes (préfixés par _)
    filtered_kwargs = _filter_kwargs(kwargs)

    result = run_pipeline(
        project_name=projet_name,
        df_train=df,
        target_col=target_col,
        enable_fe=True,
        enable_automl=False,
        **filtered_kwargs,
    )

    # Vérifications
    assert result.analyse_result is not None, "Analyse devrait être complétée"
    assert result.detected_params is not None, "Params devraient être détectés"
    assert result.feature_engineering_result is not None, "FE devrait être complété"

    print("\n--- Résultats ---")
    print(f"Task type: {result.detected_params.task_type}")
    print(f"Feature format utilisé: {result.detected_params.feature_format}")
    print(f"FE result: {result.feature_engineering_result}")

    return result


def test_pipeline_analyse_and_automl(projet_name: str, target_col: str, df: pd.DataFrame, **kwargs):
    """Test de l'analyse + AutoML (sans Feature Engineering)."""
    print("\n" + "=" * 70)
    print("TEST: Analyse + AutoML (sans FE)")
    print("=" * 70)

    from src.pipeline.pipeline_all import run_pipeline

    # Filtrer les kwargs internes (préfixés par _)
    filtered_kwargs = _filter_kwargs(kwargs)

    result = run_pipeline(
        project_name=projet_name,
        df_train=df,
        target_col=target_col,
        enable_fe=False,  # Désactive FE
        enable_automl=True,  # Active AutoML
        **filtered_kwargs,
    )

    # Vérifications
    assert result.analyse_result is not None, "Analyse devrait être complétée"
    assert result.detected_params is not None, "Params devraient être détectés"
    assert result.automl_result is not None, "AutoML devrait être complété"

    print("\n--- Résultats ---")
    print(f"Task type: {result.detected_params.task_type}")
    print(f"Metric utilisée: {result.detected_params.metric}")
    print(f"Best framework: {result.best_framework}")
    print(f"Best score: {result.best_score}")
    print(f"Output dir: {result.output_dir}")

    return result


def test_pipeline_full(projet_name: str, target_col: str, df: pd.DataFrame, **kwargs):
    """Test du pipeline complet (analyse + FE + AutoML)."""
    print("\n" + "=" * 70)
    print("TEST: Pipeline complet")
    print("=" * 70)

    from src.pipeline.pipeline_all import run_pipeline

    # Filtrer les kwargs internes (préfixés par _)
    filtered_kwargs = _filter_kwargs(kwargs)

    result = run_pipeline(
        project_name=projet_name,
        df_train=df,
        target_col=target_col,
        enable_fe=True,
        enable_automl=True,
        **filtered_kwargs,
    )

    # Vérifications
    assert result.analyse_result is not None, "Analyse devrait être complétée"
    assert result.detected_params is not None, "Params devraient être détectés"
    assert result.feature_engineering_result is not None, "FE devrait être complété"
    assert result.automl_result is not None, "AutoML devrait être complété"

    print("\n--- Résultats ---")
    print(f"Task type: {result.detected_params.task_type}")
    print(f"Metric utilisée: {result.detected_params.metric}")
    print(f"Best framework: {result.best_framework}")
    print(f"Best score: {result.best_score}")
    print(f"Output dir: {result.output_dir}")

    return result


def test_detected_params():
    """Test unitaire de la classe DetectedParams."""
    print("\n" + "=" * 70)
    print("TEST: DetectedParams")
    print("=" * 70)

    from src.pipeline.pipeline_all import DetectedParams

    # Test classification binaire équilibrée
    json_balanced = {
        "target": {
            "name": "target",
            "problem_type": "binary_classification",
            "is_imbalanced": False,
            "n_unique": 2,
        },
        "basic_stats": {
            "n_rows": 1000,
            "n_features": 10,
            "n_numeric_features": 8,
            "n_categorical_features": 2,
            "n_text_features": 0,
            "missing_cell_ratio": 0.05,
        },
    }
    params = DetectedParams(json_balanced)
    assert params.task_type == "classification"
    assert params.metric == "accuracy"
    assert params.feature_format == "tags"
    print("  Classification binaire équilibrée: OK")

    # Test classification binaire déséquilibrée
    json_imbalanced = {
        "target": {
            "name": "target",
            "problem_type": "binary_classification",
            "is_imbalanced": True,
            "imbalance_ratio": 5.0,
            "n_unique": 2,
        },
        "basic_stats": {
            "n_rows": 1000,
            "n_features": 10,
        },
    }
    params = DetectedParams(json_imbalanced)
    assert params.metric == "f1"
    print("  Classification binaire déséquilibrée: OK (metric=f1)")

    # Test regression
    json_regression = {
        "target": {
            "name": "price",
            "problem_type": "regression",
            "n_unique": 500,
        },
        "basic_stats": {
            "n_rows": 5000,
            "n_features": 20,
        },
    }
    params = DetectedParams(json_regression)
    assert params.task_type == "regression"
    assert params.metric == "rmse"
    print("  Régression: OK (metric=rmse)")

    # Test dataset complexe -> hierarchical
    json_complex = {
        "target": {"problem_type": "binary_classification", "n_unique": 2},
        "basic_stats": {
            "n_rows": 10000,
            "n_features": 60,
            "n_text_features": 3,
            "missing_cell_ratio": 0.4,
        },
    }
    params = DetectedParams(json_complex)
    assert params.feature_format == "hierarchical"
    print("  Dataset complexe: OK (format=hierarchical)")

    print("\n  Tous les tests DetectedParams passés!")


# =============================================================================
# Configuration des arguments CLI
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Crée le parser avec tous les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline ML complet: Analyse → Feature Engineering → AutoML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Analyse rapide
  python test_pipeline_all.py --dataset titanic --target Survived --analyse-only

  # Pipeline complet avec overrides
  python test_pipeline_all.py --dataset titanic --target Survived --full \\
      --override-metric f1 --automl-frameworks flaml,autogluon

Documentation complète: docs/cli_reference.md
        """,
    )

    # =========================================================================
    # 1. Paramètres principaux (dataset & projet)
    # =========================================================================
    main_group = parser.add_argument_group("Dataset & Projet")
    main_group.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="[REQUIS] Nom du dataset dans data/raw/",
    )
    main_group.add_argument(
        "--target",
        type=str,
        required=True,
        help="[REQUIS] Colonne cible à prédire",
    )
    main_group.add_argument(
        "--project",
        type=str,
        default=None,
        help="Nom du projet (défaut: nom du dataset)",
    )
    main_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Dossier racine des résultats (défaut: outputs)",
    )

    # =========================================================================
    # 2. Mode d'exécution
    # =========================================================================
    mode_group = parser.add_argument_group("Mode d'exécution")
    mode_group.add_argument(
        "--analyse-only",
        action="store_true",
        help="Analyse statistique uniquement",
    )
    mode_group.add_argument(
        "--no-automl",
        action="store_true",
        help="Analyse + Feature Engineering (sans AutoML)",
    )
    mode_group.add_argument(
        "--no-fe",
        action="store_true",
        help="Analyse + AutoML (sans Feature Engineering)",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Pipeline complet (Analyse + FE + AutoML)",
    )
    mode_group.add_argument(
        "--force-analyse",
        action="store_true",
        help="Force la regénération de l'analyse même si le JSON existe",
    )
    mode_group.add_argument(
        "--unit-tests",
        action="store_true",
        help="Lance les tests unitaires",
    )

    # =========================================================================
    # 3. Configuration Analyse
    # =========================================================================
    analyse_group = parser.add_argument_group("Configuration Analyse")
    analyse_group.add_argument(
        "--with-correlations",
        action="store_true",
        help="Active l'analyse des corrélations (Pearson, Spearman, MI, etc.)",
    )
    analyse_group.add_argument(
        "--correlation-methods",
        type=str,
        default="pearson,spearman,kendall,mutual_info",
        help="Méthodes de corrélation (défaut: pearson,spearman,kendall,mutual_info). Options: mic,phik",
    )

    # =========================================================================
    # 4. Configuration LLM
    # =========================================================================
    llm_group = parser.add_argument_group("Configuration LLM")
    llm_group.add_argument(
        "--with-llm",
        action="store_true",
        help="Active l'analyse métier LLM",
    )
    llm_group.add_argument(
        "--analyse-provider",
        type=str,
        default="openai",
        help="Provider LLM pour analyse métier (défaut: openai)",
    )
    llm_group.add_argument(
        "--analyse-model",
        type=str,
        default="gpt-4o-mini",
        help="Modèle pour analyse métier (défaut: gpt-4o-mini)",
    )
    llm_group.add_argument(
        "--llmfe-model",
        type=str,
        default="gpt-3.5-turbo",
        help="Modèle pour Feature Engineering (défaut: gpt-3.5-turbo)",
    )

    # =========================================================================
    # 5. Overrides (forcer des valeurs)
    # =========================================================================
    override_group = parser.add_argument_group("Overrides (forcer des valeurs)")
    override_group.add_argument(
        "--override-task-type",
        type=str,
        choices=["classification", "regression"],
        help="Force le type de tâche",
    )
    override_group.add_argument(
        "--override-metric",
        type=str,
        help="Force la métrique (f1, accuracy, rmse, roc_auc...)",
    )
    override_group.add_argument(
        "--override-feature-format",
        type=str,
        choices=["basic", "tags", "hierarchical"],
        help="Force le format de features",
    )
    override_group.add_argument(
        "--override-max-samples",
        type=int,
        default=3,
        help="Nombre d'itérations LLMFE (défaut: 3)",
    )
    override_group.add_argument(
        "--override-time-budget",
        type=int,
        default=60,
        help="Budget temps AutoML en secondes (défaut: 60)",
    )

    # =========================================================================
    # 6. Configuration AutoML
    # =========================================================================
    automl_group = parser.add_argument_group("Configuration AutoML")
    automl_group.add_argument(
        "--automl-frameworks",
        type=str,
        default="flaml,autogluon",
        help="Frameworks séparés par virgule (défaut: flaml,autogluon)",
    )

    # FLAML
    automl_group.add_argument(
        "--flaml-time-budget",
        type=int,
        help="Budget temps FLAML en secondes",
    )
    automl_group.add_argument(
        "--flaml-metric",
        type=str,
        help="Métrique FLAML",
    )

    # AutoGluon
    automl_group.add_argument(
        "--autogluon-presets",
        type=str,
        default="medium_quality_faster_train",
        help="Preset AutoGluon (défaut: medium_quality_faster_train)",
    )
    automl_group.add_argument(
        "--autogluon-time-budget",
        type=int,
        help="Budget temps AutoGluon en secondes",
    )

    # TPOT
    automl_group.add_argument(
        "--tpot-generations",
        type=int,
        default=7,
        help="Nombre de générations TPOT (défaut: 7)",
    )
    automl_group.add_argument(
        "--tpot-population-size",
        type=int,
        default=25,
        help="Taille de population TPOT (défaut: 25)",
    )
    automl_group.add_argument(
        "--tpot-cv",
        type=int,
        default=5,
        help="Folds cross-validation TPOT (défaut: 5)",
    )

    # H2O
    automl_group.add_argument(
        "--h2o-time-budget",
        type=int,
        help="Budget temps H2O en secondes",
    )
    automl_group.add_argument(
        "--h2o-verbosity",
        type=str,
        default="info",
        choices=["debug", "info", "warn"],
        help="Niveau de log H2O (défaut: info)",
    )
    automl_group.add_argument(
        "--h2o-no-mojo",
        action="store_true",
        help="Désactive l'export MOJO H2O",
    )

    # =========================================================================
    # 7. Seuils d'analyse (avancé)
    # =========================================================================
    threshold_group = parser.add_argument_group("Seuils d'analyse (avancé)")
    threshold_group.add_argument(
        "--high-cardinality-threshold",
        type=int,
        default=50,
        help="Seuil haute cardinalité (défaut: 50)",
    )
    threshold_group.add_argument(
        "--high-missing-threshold",
        type=float,
        default=0.3,
        help="Seuil valeurs manquantes (défaut: 0.3)",
    )
    threshold_group.add_argument(
        "--strong-corr-threshold",
        type=float,
        default=0.97,
        help="Seuil corrélation pour leakage (défaut: 0.97)",
    )
    threshold_group.add_argument(
        "--text-unique-ratio",
        type=float,
        default=0.5,
        help="Ratio unicité pour texte (défaut: 0.5)",
    )
    threshold_group.add_argument(
        "--id-unique-ratio",
        type=float,
        default=0.9,
        help="Ratio unicité pour ID (défaut: 0.9)",
    )

    # =========================================================================
    # 8. Performance & Parallélisation
    # =========================================================================
    perf_group = parser.add_argument_group("Performance & Parallélisation")
    perf_group.add_argument(
        "--num-samplers",
        type=int,
        default=1,
        help="Samplers parallèles LLMFE (défaut: 1)",
    )
    perf_group.add_argument(
        "--num-evaluators",
        type=int,
        default=1,
        help="Évaluateurs parallèles LLMFE (défaut: 1)",
    )
    perf_group.add_argument(
        "--evaluate-timeout",
        type=int,
        default=30,
        help="Timeout évaluation code en secondes (défaut: 30)",
    )
    perf_group.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="CPUs pour TPOT, -1 = tous (défaut: -1)",
    )

    return parser


def build_kwargs_from_args(args) -> dict:
    """Construit le dictionnaire kwargs à partir des arguments CLI."""
    kwargs = {}

    # Output directory
    if args.output_dir != "outputs":
        kwargs["output_dir"] = args.output_dir

    # Force analyse (regénérer même si JSON existe)
    if args.force_analyse:
        kwargs["force_analyse"] = True

    # Corrélations
    if args.with_correlations:
        kwargs["with_correlations"] = True
        kwargs["correlation_methods"] = parse_list(args.correlation_methods)

    # LLM configuration
    if args.with_llm:
        kwargs["analyse_only_stats"] = False
    if args.analyse_provider != "openai":
        kwargs["analyse_provider"] = args.analyse_provider
    if args.analyse_model != "gpt-4o-mini":
        kwargs["analyse_model"] = args.analyse_model
    if args.llmfe_model != "gpt-3.5-turbo":
        kwargs["llmfe_model"] = args.llmfe_model

    # Overrides
    if args.override_task_type:
        kwargs["override_task_type"] = args.override_task_type
    if args.override_metric:
        kwargs["override_metric"] = args.override_metric
    if args.override_feature_format:
        kwargs["override_feature_format"] = args.override_feature_format
    if args.override_max_samples:
        kwargs["override_max_samples"] = args.override_max_samples
    if args.override_time_budget:
        kwargs["override_time_budget"] = args.override_time_budget

    # AutoML frameworks
    kwargs["automl_frameworks"] = parse_list(args.automl_frameworks)

    # AutoML specific configs (stored for later use)
    kwargs["_automl_config"] = {
        "flaml": {
            "time_budget": args.flaml_time_budget,
            "metric": args.flaml_metric,
        },
        "autogluon": {
            "presets": args.autogluon_presets,
            "time_budget": args.autogluon_time_budget,
        },
        "tpot": {
            "generations": args.tpot_generations,
            "population_size": args.tpot_population_size,
            "cv": args.tpot_cv,
        },
        "h2o": {
            "time_budget": args.h2o_time_budget,
            "verbosity": args.h2o_verbosity,
            "save_mojo": not args.h2o_no_mojo,
        },
    }

    # Analysis thresholds (for FEAnalysisConfig)
    kwargs["_analysis_config"] = {
        "high_cardinality_threshold": args.high_cardinality_threshold,
        "high_missing_threshold": args.high_missing_threshold,
        "strong_corr_threshold": args.strong_corr_threshold,
        "text_unique_ratio_threshold": args.text_unique_ratio,
        "id_unique_ratio_threshold": args.id_unique_ratio,
    }

    # Performance
    kwargs["_performance_config"] = {
        "num_samplers": args.num_samplers,
        "num_evaluators": args.num_evaluators,
        "evaluate_timeout_seconds": args.evaluate_timeout,
        "n_jobs": args.n_jobs,
    }

    return kwargs


# =============================================================================
# Main
# =============================================================================


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Tests unitaires (ne nécessite pas dataset/target)
    if args.unit_tests:
        test_detected_params()
        return

    # Charger les données
    df = load_data(args.dataset)

    # Paramètres du projet (défaut: nom du dataset)
    projet_name = args.project if args.project else args.dataset
    target_col = args.target

    # Construire les kwargs
    kwargs = build_kwargs_from_args(args)

    # Afficher la configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"  Dataset:       {args.dataset}")
    print(f"  Project:       {projet_name}")
    print(f"  Target:        {target_col}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Force analyse: {args.force_analyse}")
    print(f"  With LLM:      {args.with_llm}")
    print(f"  Correlations:  {args.with_correlations}")
    if args.with_correlations:
        print(f"  Corr methods:  {args.correlation_methods}")
    print(f"  Frameworks:    {kwargs.get('automl_frameworks', [])}")

    # Exécuter le test approprié
    if args.analyse_only:
        test_pipeline_analyse_only(projet_name, target_col, df, **kwargs)
    elif args.no_automl:
        test_pipeline_analyse_and_fe(projet_name, target_col, df, **kwargs)
    elif args.no_fe:
        test_pipeline_analyse_and_automl(projet_name, target_col, df, **kwargs)
    elif args.full:
        test_pipeline_full(projet_name, target_col, df, **kwargs)
    else:
        # Par défaut: test analyse seulement (le plus rapide)
        print("\nAucune option spécifiée. Exécution du test d'analyse seule.")
        print("Utilisez --full pour le test complet, --no-automl pour analyse+FE")
        test_pipeline_analyse_only(projet_name, target_col, df, **kwargs)


if __name__ == "__main__":
    main()
