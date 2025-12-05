#!/usr/bin/env python
# tests/integration/test_dfs.py
"""
Test d'intégration pour Deep Feature Synthesis.

Usage:
    # Test avec une configuration spécifique
    python tests/integration/test_dfs.py --dataset titanic --target Survived --pipeline minimal

    # Test avec toutes les configurations (comparaison)
    python tests/integration/test_dfs.py --dataset titanic --target Survived --all-pipelines

    # Liste des pipelines disponibles
    python tests/integration/test_dfs.py --list-pipelines

    # Test personnalisé
    python tests/integration/test_dfs.py --dataset titanic --target Survived --max-depth 3 --top-k 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Bootstrapping: add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Charger le .env
from dotenv import load_dotenv

env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"[CONFIG] Fichier .env chargé depuis {env_path}")

import pandas as pd

from src.core.config import settings
from src.feature_engineering.dfs import DFSConfig, DFSRunner, run_dfs

# =============================================================================
# PIPELINES DFS PRÉDÉFINIS POUR LE MÉTA-LEARNING
# =============================================================================

DFS_PIPELINES: dict[str, DFSConfig] = {
    # --- Pipelines minimalistes (rapides, peu de features) ---
    "minimal": DFSConfig(
        max_depth=1,
        agg_primitives=["mean", "sum", "count"],
        trans_primitives=["is_null", "absolute"],
        feature_selection=True,
        selection_method="importance",
        selection_threshold=0.02,
        top_k_features=15,
        eval_models=["xgboost"],
        verbose=False,
    ),
    "minimal_strict": DFSConfig(
        max_depth=1,
        agg_primitives=["mean", "sum", "count", "max", "min"],
        trans_primitives=["is_null"],
        feature_selection=True,
        selection_method="hybrid",
        selection_threshold=0.05,
        top_k_features=10,
        eval_models=["xgboost"],
        verbose=False,
    ),
    # --- Pipelines standard (équilibrés) ---
    "standard": DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "sum", "count", "max", "min", "std", "median"],
        trans_primitives=["is_null", "absolute", "percentile"],
        feature_selection=True,
        selection_method="hybrid",
        selection_threshold=0.01,
        top_k_features=30,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
    "standard_importance": DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "sum", "count", "max", "min", "std", "median", "mode"],
        trans_primitives=["is_null", "absolute", "percentile"],
        feature_selection=True,
        selection_method="importance",
        selection_threshold=0.005,
        top_k_features=40,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
    "standard_correlation": DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "sum", "count", "max", "min", "std"],
        trans_primitives=["is_null", "absolute"],
        feature_selection=True,
        selection_method="correlation",
        correlation_threshold=0.90,
        top_k_features=35,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
    # --- Pipelines avec features temporelles (pour datasets avec datetime) ---
    "datetime_basic": DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "sum", "count", "max", "min", "first", "last"],
        trans_primitives=["year", "month", "day", "weekday", "is_null"],
        feature_selection=True,
        selection_method="hybrid",
        selection_threshold=0.01,
        top_k_features=40,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
    "datetime_advanced": DFSConfig(
        max_depth=2,
        agg_primitives=[
            "mean",
            "sum",
            "count",
            "max",
            "min",
            "std",
            "first",
            "last",
            "time_since_last",
        ],
        trans_primitives=[
            "year",
            "month",
            "day",
            "weekday",
            "hour",
            "is_weekend",
            "week",
            "quarter",
            "is_null",
        ],
        feature_selection=True,
        selection_method="hybrid",
        selection_threshold=0.01,
        top_k_features=50,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
    # --- Pipelines exhaustifs (maximum de features, plus lent) ---
    "exhaustive": DFSConfig(
        max_depth=2,
        agg_primitives=[
            "mean",
            "sum",
            "count",
            "max",
            "min",
            "std",
            "median",
            "mode",
            "num_unique",
            "skew",
        ],
        trans_primitives=["is_null", "absolute", "percentile", "cum_sum", "cum_mean"],
        feature_selection=True,
        selection_method="hybrid",
        selection_threshold=0.005,
        top_k_features=75,
        eval_models=["xgboost", "lightgbm", "randomforest"],
        verbose=False,
    ),
    "exhaustive_rfe": DFSConfig(
        max_depth=2,
        agg_primitives=[
            "mean",
            "sum",
            "count",
            "max",
            "min",
            "std",
            "median",
            "mode",
            "num_unique",
        ],
        trans_primitives=["is_null", "absolute", "percentile"],
        feature_selection=True,
        selection_method="rfe",
        top_k_features=50,
        eval_models=["xgboost", "lightgbm", "randomforest"],
        verbose=False,
    ),
    # --- Pipelines profonds (depth=3, pour données relationnelles) ---
    "deep_standard": DFSConfig(
        max_depth=3,
        agg_primitives=["mean", "sum", "count", "max", "min", "std"],
        trans_primitives=["is_null", "absolute"],
        feature_selection=True,
        selection_method="hybrid",
        selection_threshold=0.01,
        top_k_features=60,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
    "deep_exhaustive": DFSConfig(
        max_depth=3,
        agg_primitives=[
            "mean",
            "sum",
            "count",
            "max",
            "min",
            "std",
            "median",
            "mode",
            "num_unique",
            "skew",
            "entropy",
        ],
        trans_primitives=["is_null", "absolute", "percentile", "cum_sum", "diff"],
        feature_selection=True,
        selection_method="rfe",
        top_k_features=100,
        eval_models=["xgboost", "lightgbm", "randomforest"],
        verbose=False,
    ),
    # --- Pipelines sans sélection (pour benchmark) ---
    "no_selection": DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "sum", "count", "max", "min", "std", "median"],
        trans_primitives=["is_null", "absolute", "percentile"],
        feature_selection=False,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
    # --- Pipelines spécialisés par type de données ---
    "numeric_only": DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "sum", "count", "max", "min", "std", "median", "skew"],
        trans_primitives=["absolute", "percentile", "cum_sum", "cum_mean", "cum_max", "diff"],
        feature_selection=True,
        selection_method="importance",
        selection_threshold=0.01,
        top_k_features=50,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
    "categorical_focus": DFSConfig(
        max_depth=2,
        agg_primitives=["count", "mode", "num_unique", "percent_true"],
        trans_primitives=["is_null"],
        feature_selection=True,
        selection_method="importance",
        selection_threshold=0.01,
        top_k_features=30,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
    # ==========================================================================
    # PIPELINES AVEC RELATIONS SYNTHÉTIQUES (pour single-table data)
    # ==========================================================================
    # Ces pipelines créent des "pseudo-tables" en groupant par colonnes
    # catégorielles, ce qui permet à DFS d'utiliser les primitives d'agrégation
    # même sur des données single-table (comme Titanic).
    #
    # Exemple: Pour Titanic, on crée des tables:
    # - Pclass_groups (3 classes)
    # - Sex_groups (2 sexes)
    # - Embarked_groups (3 ports)
    #
    # DFS peut alors calculer: MEAN(Pclass_groups.Age), STD(Sex_groups.Fare), etc.
    # ==========================================================================
    "synthetic_minimal": DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "std", "min", "max", "count"],
        trans_primitives=["is_null"],
        feature_selection=True,
        selection_method="importance",
        selection_threshold=0.01,
        top_k_features=20,
        create_synthetic_relations=True,
        max_synthetic_tables=3,
        eval_models=["xgboost"],
        verbose=False,
    ),
    "synthetic_standard": DFSConfig(
        max_depth=2,
        agg_primitives=["mean", "std", "min", "max", "count", "median", "num_unique"],
        trans_primitives=["is_null", "absolute", "percentile"],
        feature_selection=True,
        selection_method="hybrid",
        selection_threshold=0.01,
        top_k_features=40,
        create_synthetic_relations=True,
        max_synthetic_tables=5,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
    "synthetic_exhaustive": DFSConfig(
        max_depth=2,
        agg_primitives=[
            "mean",
            "std",
            "min",
            "max",
            "count",
            "median",
            "mode",
            "num_unique",
            "skew",
        ],
        trans_primitives=["is_null", "absolute", "percentile", "cum_sum"],
        feature_selection=True,
        selection_method="hybrid",
        selection_threshold=0.005,
        top_k_features=75,
        create_synthetic_relations=True,
        max_synthetic_tables=6,
        eval_models=["xgboost", "lightgbm", "randomforest"],
        verbose=False,
    ),
    "synthetic_deep": DFSConfig(
        max_depth=3,
        agg_primitives=["mean", "std", "min", "max", "count", "median", "num_unique"],
        trans_primitives=["is_null", "absolute"],
        feature_selection=True,
        selection_method="rfe",
        top_k_features=60,
        create_synthetic_relations=True,
        max_synthetic_tables=5,
        eval_models=["xgboost", "lightgbm"],
        verbose=False,
    ),
}


def get_pipeline(name: str) -> DFSConfig:
    """Récupère un pipeline par son nom."""
    if name not in DFS_PIPELINES:
        raise ValueError(f"Pipeline inconnu: {name}. " f"Disponibles: {list(DFS_PIPELINES.keys())}")
    return DFS_PIPELINES[name]


def list_pipelines() -> None:
    """Affiche tous les pipelines disponibles."""
    print("\n" + "=" * 80)
    print("  PIPELINES DFS DISPONIBLES")
    print("=" * 80)

    categories = {
        "Minimalistes (rapides)": ["minimal", "minimal_strict"],
        "Standard (équilibrés)": ["standard", "standard_importance", "standard_correlation"],
        "Datetime (temporel)": ["datetime_basic", "datetime_advanced"],
        "Exhaustifs (complets)": ["exhaustive", "exhaustive_rfe"],
        "Profonds (depth=3)": ["deep_standard", "deep_exhaustive"],
        "Spécialisés": ["no_selection", "numeric_only", "categorical_focus"],
        "Synthétiques (single-table)": [
            "synthetic_minimal",
            "synthetic_standard",
            "synthetic_exhaustive",
            "synthetic_deep",
        ],
    }

    for category, pipeline_names in categories.items():
        print(f"\n  {category}:")
        print("  " + "-" * 40)
        for name in pipeline_names:
            config = DFS_PIPELINES[name]
            print(f"    • {name}")
            print(
                f"        depth={config.max_depth}, "
                f"selection={config.selection_method if config.feature_selection else 'None'}, "
                f"top_k={config.top_k_features}"
            )

    print("\n" + "=" * 80)
    print("  Usage: python test_dfs.py --dataset X --target Y --pipeline <nom>")
    print("         python test_dfs.py --dataset X --target Y --all-pipelines")
    print("=" * 80 + "\n")


@dataclass
class PipelineResult:
    """Résultat d'un pipeline DFS."""

    pipeline_name: str
    n_features_generated: int
    n_features_selected: int
    initial_score: float
    final_score: float
    improvement_pct: float
    execution_time: float
    scores_by_model: dict[str, float] = field(default_factory=dict)


def run_all_pipelines(
    df_train: pd.DataFrame,
    target_col: str,
    project_name: str,
    is_regression: bool = False,
    pipelines: list[str] | None = None,
    verbose: bool = True,
) -> list[PipelineResult]:
    """
    Exécute tous les pipelines DFS sur un dataset.

    Args:
        df_train: DataFrame d'entraînement
        target_col: Colonne cible
        project_name: Nom du projet
        is_regression: Type de tâche
        pipelines: Liste des pipelines à tester (None = tous)
        verbose: Afficher les logs

    Returns:
        Liste des résultats par pipeline
    """
    if pipelines is None:
        pipelines = list(DFS_PIPELINES.keys())

    results: list[PipelineResult] = []

    print("\n" + "=" * 80)
    print("  COMPARAISON DES PIPELINES DFS")
    print("=" * 80)
    print(f"  Dataset: {df_train.shape[0]} lignes × {df_train.shape[1]} colonnes")
    print(f"  Target: {target_col}")
    print(f"  Pipelines à tester: {len(pipelines)}")
    print("=" * 80)

    for i, pipeline_name in enumerate(pipelines, 1):
        print(f"\n[{i}/{len(pipelines)}] Pipeline: {pipeline_name}")
        print("-" * 50)

        try:
            config = get_pipeline(pipeline_name)
            # Activer verbose pour ce run si demandé
            config.verbose = verbose

            runner = DFSRunner(
                project_name=f"{project_name}_{pipeline_name}",
                config=config,
            )

            start_time = time.time()
            result = runner.run(
                df_train=df_train,
                target_col=target_col,
                is_regression=is_regression,
            )
            exec_time = time.time() - start_time

            pipeline_result = PipelineResult(
                pipeline_name=pipeline_name,
                n_features_generated=result.n_features_generated,
                n_features_selected=result.n_features_selected,
                initial_score=result.initial_score,
                final_score=result.final_score,
                improvement_pct=result.improvement_pct,
                execution_time=exec_time,
                scores_by_model=result.scores_by_model,
            )
            results.append(pipeline_result)

            print(f"  ✓ Features: {result.n_features_generated} → {result.n_features_selected}")
            print(
                f"  ✓ Score: {result.initial_score:.4f} → {result.final_score:.4f} ({result.improvement_pct:+.2f}%)"
            )
            print(f"  ✓ Temps: {exec_time:.1f}s")

        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            continue

    return results


def print_comparison_summary(results: list[PipelineResult]) -> None:
    """Affiche le résumé de la comparaison."""
    if not results:
        print("\nAucun résultat à afficher.")
        return

    # Trier par score final décroissant
    sorted_results = sorted(results, key=lambda r: r.final_score, reverse=True)

    print("\n" + "=" * 100)
    print("  RÉSUMÉ DE LA COMPARAISON")
    print("=" * 100)
    print(
        f"{'Pipeline':<25} {'Features':<12} {'Score Init':<12} {'Score Final':<12} {'Δ %':<10} {'Temps':<8}"
    )
    print("-" * 100)

    for i, r in enumerate(sorted_results):
        marker = "★" if i == 0 else " "
        features = f"{r.n_features_generated}→{r.n_features_selected}"
        print(
            f"{marker} {r.pipeline_name:<23} "
            f"{features:<12} "
            f"{r.initial_score:<12.4f} "
            f"{r.final_score:<12.4f} "
            f"{r.improvement_pct:>+8.2f}% "
            f"{r.execution_time:>6.1f}s"
        )

    print("-" * 100)

    # Meilleur pipeline
    best = sorted_results[0]
    print(f"\n★ MEILLEUR PIPELINE: {best.pipeline_name}")
    print(f"  Score final: {best.final_score:.4f}")
    print(f"  Amélioration: {best.improvement_pct:+.2f}%")
    print(f"  Features: {best.n_features_selected}")

    if best.scores_by_model:
        print("  Scores par modèle:")
        for model, score in best.scores_by_model.items():
            print(f"    - {model}: {score:.4f}")

    print("=" * 100)


def save_comparison_results(
    results: list[PipelineResult],
    project_name: str,
    dataset_name: str,
    target_col: str,
) -> Path:
    """Sauvegarde les résultats de la comparaison."""
    output_dir = Path(settings.output_dir) / project_name / "dfs_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convertir en dict
    data = {
        "project_name": project_name,
        "dataset_name": dataset_name,
        "target_col": target_col,
        "timestamp": datetime.now().isoformat(),
        "n_pipelines": len(results),
        "results": [
            {
                "pipeline_name": r.pipeline_name,
                "n_features_generated": r.n_features_generated,
                "n_features_selected": r.n_features_selected,
                "initial_score": r.initial_score,
                "final_score": r.final_score,
                "improvement_pct": r.improvement_pct,
                "execution_time": r.execution_time,
                "scores_by_model": r.scores_by_model,
            }
            for r in results
        ],
        "best_pipeline": max(results, key=lambda r: r.final_score).pipeline_name
        if results
        else None,
    }

    filepath = output_dir / "comparison_results.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Résultats sauvegardés: {filepath}")
    return filepath


def load_dataset(dataset_name: str, target_col: str) -> pd.DataFrame:
    """Charge un dataset depuis data/raw/."""
    data_dir = Path(settings.data_dir) / dataset_name

    possible_files = [
        data_dir / "train.csv",
        data_dir / f"{dataset_name}.csv",
        data_dir / "data.csv",
    ]

    for file_path in possible_files:
        if file_path.exists():
            print(f"📂 Chargement du dataset: {file_path}")
            df = pd.read_csv(file_path)
            print(f"   Shape: {df.shape}")
            print(f"   Target: {target_col}")
            return df

    available_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' non trouvé.\n"
        f"Fichiers disponibles: {[f.name for f in available_files]}"
    )


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Test Deep Feature Synthesis (DFS) avec pipelines prédéfinis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Lister les pipelines disponibles
  python test_dfs.py --list-pipelines

  # Tester un pipeline spécifique
  python test_dfs.py --dataset titanic --target Survived --pipeline standard

  # Comparer tous les pipelines
  python test_dfs.py --dataset titanic --target Survived --all-pipelines

  # Comparer certains pipelines
  python test_dfs.py --dataset titanic --target Survived --pipelines minimal,standard,exhaustive

  # Configuration personnalisée
  python test_dfs.py --dataset titanic --target Survived --max-depth 3 --top-k 50
        """,
    )

    # Actions principales
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="Afficher tous les pipelines disponibles",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        help="Nom du dataset (dossier dans data/raw/)",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Nom de la colonne cible",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Nom du projet (défaut: {dataset}_dfs)",
    )
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Problème de régression (défaut: classification)",
    )

    # Pipelines
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        help="Nom du pipeline à utiliser (voir --list-pipelines)",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default=None,
        help="Liste de pipelines séparés par virgule (ex: minimal,standard,exhaustive)",
    )
    parser.add_argument(
        "--all-pipelines",
        action="store_true",
        help="Tester tous les pipelines disponibles",
    )

    # Configuration personnalisée (override)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Profondeur maximale DFS (override)",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default=None,
        choices=["importance", "correlation", "rfe", "hybrid"],
        help="Méthode de sélection (override)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Nombre maximum de features à garder (override)",
    )
    parser.add_argument(
        "--no-selection",
        action="store_true",
        help="Désactiver la sélection automatique des features",
    )

    # Options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mode silencieux",
    )

    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()

    # Lister les pipelines
    if args.list_pipelines:
        list_pipelines()
        return

    # Vérifier les arguments requis
    if not args.dataset or not args.target:
        print("❌ Erreur: --dataset et --target sont requis")
        print("   Utilisez --list-pipelines pour voir les pipelines disponibles")
        sys.exit(1)

    project_name = args.project or f"{args.dataset}_dfs"

    # Charger le dataset
    try:
        df_train = load_dataset(args.dataset, args.target)
    except FileNotFoundError as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)

    # Mode: tous les pipelines
    if args.all_pipelines:
        results = run_all_pipelines(
            df_train=df_train,
            target_col=args.target,
            project_name=project_name,
            is_regression=args.regression,
            verbose=not args.quiet,
        )
        print_comparison_summary(results)
        save_comparison_results(results, project_name, args.dataset, args.target)
        return

    # Mode: plusieurs pipelines spécifiques
    if args.pipelines:
        pipeline_list = [p.strip() for p in args.pipelines.split(",")]
        results = run_all_pipelines(
            df_train=df_train,
            target_col=args.target,
            project_name=project_name,
            is_regression=args.regression,
            pipelines=pipeline_list,
            verbose=not args.quiet,
        )
        print_comparison_summary(results)
        save_comparison_results(results, project_name, args.dataset, args.target)
        return

    # Mode: un seul pipeline ou configuration personnalisée
    if args.pipeline:
        config = get_pipeline(args.pipeline)
    else:
        # Configuration par défaut ou personnalisée
        config = DFSConfig(
            max_depth=args.max_depth or 2,
            feature_selection=not args.no_selection,
            selection_method=args.selection_method or "hybrid",
            top_k_features=args.top_k,
            eval_models=["xgboost", "lightgbm", "randomforest"],
            verbose=not args.quiet,
        )

    # Appliquer les overrides si spécifiés
    if args.max_depth is not None:
        config.max_depth = args.max_depth
    if args.selection_method is not None:
        config.selection_method = args.selection_method
    if args.top_k is not None:
        config.top_k_features = args.top_k
    if args.no_selection:
        config.feature_selection = False

    config.verbose = not args.quiet

    # Afficher la configuration
    print("\n" + "=" * 70)
    print("  🔬 TEST DEEP FEATURE SYNTHESIS (DFS)")
    print("=" * 70)
    print(f"  Dataset      : {args.dataset}")
    print(f"  Target       : {args.target}")
    print(f"  Project      : {project_name}")
    print(f"  Type         : {'Régression' if args.regression else 'Classification'}")
    print(f"  Pipeline     : {args.pipeline or 'custom'}")
    print(f"  Max depth    : {config.max_depth}")
    print(
        f"  Sélection    : {config.selection_method if config.feature_selection else 'Désactivée'}"
    )
    print(f"  Top K        : {config.top_k_features or 'Illimité'}")
    print("=" * 70)

    # Exécuter
    runner = DFSRunner(project_name=project_name, config=config)
    result = runner.run(
        df_train=df_train,
        target_col=args.target,
        is_regression=args.regression,
    )

    # Afficher les résultats
    print("\n" + "=" * 70)
    print("  📊 RÉSULTATS")
    print("=" * 70)
    print(f"  Features générées      : {result.n_features_generated}")
    print(f"  Features sélectionnées : {result.n_features_selected}")
    print(f"  Score initial          : {result.initial_score:.4f}")
    print(f"  Score final            : {result.final_score:.4f}")
    print(f"  Amélioration           : {result.improvement_pct:+.2f}%")
    print(f"  Temps total            : {result.execution_time_seconds:.1f}s")

    if result.scores_by_model:
        print("\n  Scores par modèle:")
        for model, score in result.scores_by_model.items():
            print(f"    - {model}: {score:.4f}")

    print("\n  Top 10 features:")
    for i, feat in enumerate(result.feature_names[:10], 1):
        print(f"    {i}. {feat}")

    if len(result.feature_names) > 10:
        print(f"    ... et {len(result.feature_names) - 10} autres")

    print("=" * 70)

    # Chemin des outputs
    print("\n" + "─" * 70)
    print("📁 Fichiers générés:")
    print(f"   - outputs/{project_name}/feature_engineering/dfs/train_dfs.parquet")
    print(f"   - outputs/{project_name}/feature_engineering/dfs/dfs_report.json")
    print(f"   - outputs/{project_name}/feature_engineering/dfs/feature_definitions.json")
    print("─" * 70 + "\n")


if __name__ == "__main__":
    main()
