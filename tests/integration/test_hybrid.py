#!/usr/bin/env python
# tests/integration/test_hybrid.py
"""
Test d'intégration pour le Feature Engineering Hybride (LLMFE + DFS).

Usage:
    # Test complet hybride (LLMFE + DFS)
    python tests/integration/test_hybrid.py --dataset titanic --target Survived

    # Test avec configuration spécifique
    python tests/integration/test_hybrid.py --dataset titanic --target Survived --config fast

    # Test DFS seul (si pas de clé API LLM)
    python tests/integration/test_hybrid.py --dataset titanic --target Survived --config dfs_only

    # Test LLMFE seul
    python tests/integration/test_hybrid.py --dataset titanic --target Survived --config llmfe_only

    # Comparer les configurations
    python tests/integration/test_hybrid.py --dataset titanic --target Survived --compare-configs
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Bootstrapping: add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configuration environnement
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Charger le .env
from dotenv import load_dotenv

env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"[CONFIG] Fichier .env chargé depuis {env_path}")

import pandas as pd

from src.core.config import settings
from src.feature_engineering.hybrid import (
    HYBRID_CONFIGS,
    HybridConfig,
    HybridFeatureEngineer,
    run_hybrid_fe,
)


def check_api_key(config_name: str) -> tuple[bool, str]:
    """
    Vérifie si une clé API est disponible pour LLMFE.

    Returns:
        Tuple (api_available, message)
    """
    # Si config DFS only, pas besoin d'API
    if config_name == "dfs_only":
        return True, "Mode DFS seul - pas de clé API requise"

    # Vérifier OpenAI
    if settings.is_configured("openai"):
        os.environ["API_KEY"] = settings.openai_api_key
        return True, "OpenAI API configurée"

    # Vérifier Anthropic
    if settings.is_configured("anthropic"):
        os.environ["API_KEY"] = settings.anthropic_api_key
        return True, "Anthropic API configurée"

    # Aucune clé trouvée
    return False, "Aucune clé API trouvée (OPENAI_API_KEY ou ANTHROPIC_API_KEY)"


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
            print(f"Chargement du dataset: {file_path}")
            df = pd.read_csv(file_path)
            print(f"   Shape: {df.shape}")
            print(f"   Target: {target_col}")
            return df

    available_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' non trouvé.\n"
        f"Fichiers disponibles: {[f.name for f in available_files]}"
    )


def list_configs() -> None:
    """Affiche toutes les configurations disponibles."""
    print("\n" + "=" * 70)
    print("  CONFIGURATIONS HYBRIDES DISPONIBLES")
    print("=" * 70)

    for name, config in HYBRID_CONFIGS.items():
        llmfe_status = "ON" if config.enable_llmfe else "OFF"
        dfs_status = "ON" if config.enable_dfs else "OFF"
        print(f"\n  {name}:")
        print(f"      LLMFE: {llmfe_status} (max_iter={config.llmfe_max_iterations})")
        print(f"      DFS: {dfs_status}")
        print(f"      Max features: {config.max_features or 'Illimité'}")
        print(f"      Priorité: {config.feature_priority}")

    print("\n" + "=" * 70)


def compare_configs(
    df_train: pd.DataFrame,
    target_col: str,
    project_name: str,
    is_regression: bool,
    configs_to_test: list[str] | None = None,
) -> None:
    """Compare plusieurs configurations hybrides."""
    if configs_to_test is None:
        # Tester toutes les configs sauf celles nécessitant LLM si pas de clé
        configs_to_test = list(HYBRID_CONFIGS.keys())

    results = []

    print("\n" + "=" * 80)
    print("  COMPARAISON DES CONFIGURATIONS HYBRIDES")
    print("=" * 80)
    print(f"  Dataset: {df_train.shape[0]} lignes × {df_train.shape[1]} colonnes")
    print(f"  Target: {target_col}")
    print(f"  Configs à tester: {len(configs_to_test)}")
    print("=" * 80)

    for i, config_name in enumerate(configs_to_test, 1):
        print(f"\n[{i}/{len(configs_to_test)}] Configuration: {config_name}")
        print("-" * 50)

        try:
            start_time = time.time()

            df_transformed, result = run_hybrid_fe(
                df_train=df_train,
                target_col=target_col,
                project_name=f"{project_name}_{config_name}",
                config=config_name,
                is_regression=is_regression,
                verbose=False,
            )

            exec_time = time.time() - start_time

            results.append(
                {
                    "config": config_name,
                    "n_original": result.n_original,
                    "n_llmfe": result.n_llmfe_added,
                    "n_dfs": result.n_dfs_added,
                    "n_final": result.n_final,
                    "baseline": result.baseline_score,
                    "final_score": result.final_score,
                    "improvement": result.improvement_pct,
                    "time": exec_time,
                }
            )

            print(
                f"  Features: {result.n_original} + {result.n_llmfe_added} LLMFE + {result.n_dfs_added} DFS → {result.n_final} final"
            )
            print(
                f"  Score: {result.baseline_score:.4f} → {result.final_score:.4f} ({result.improvement_pct:+.2f}%)"
            )
            print(f"  Temps: {exec_time:.1f}s")

        except Exception as e:
            print(f"  Erreur: {e}")
            continue

    # Résumé
    if results:
        print("\n" + "=" * 100)
        print("  RÉSUMÉ DE LA COMPARAISON")
        print("=" * 100)
        print(
            f"{'Config':<20} {'Original':<10} {'LLMFE':<8} {'DFS':<8} {'Final':<8} {'Baseline':<10} {'Score':<10} {'Δ %':<10} {'Temps':<8}"
        )
        print("-" * 100)

        # Trier par score final
        sorted_results = sorted(results, key=lambda r: r["final_score"], reverse=True)

        for i, r in enumerate(sorted_results):
            marker = "★" if i == 0 else " "
            print(
                f"{marker} {r['config']:<18} "
                f"{r['n_original']:<10} "
                f"{r['n_llmfe']:<8} "
                f"{r['n_dfs']:<8} "
                f"{r['n_final']:<8} "
                f"{r['baseline']:<10.4f} "
                f"{r['final_score']:<10.4f} "
                f"{r['improvement']:>+8.2f}% "
                f"{r['time']:>6.1f}s"
            )

        print("-" * 100)
        best = sorted_results[0]
        print(f"\n★ MEILLEURE CONFIG: {best['config']}")
        print(f"  Score final: {best['final_score']:.4f}")
        print(f"  Amélioration: {best['improvement']:+.2f}%")
        print("=" * 100)


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Test Feature Engineering Hybride (LLMFE + DFS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Test par défaut (hybride)
  python test_hybrid.py --dataset titanic --target Survived

  # Test avec config spécifique
  python test_hybrid.py --dataset titanic --target Survived --config fast

  # DFS seul (sans LLM)
  python test_hybrid.py --dataset titanic --target Survived --config dfs_only

  # Comparer les configs
  python test_hybrid.py --dataset titanic --target Survived --compare-configs

  # Lister les configs
  python test_hybrid.py --list-configs
        """,
    )

    # Actions
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Afficher les configurations disponibles",
    )
    parser.add_argument(
        "--compare-configs",
        action="store_true",
        help="Comparer toutes les configurations",
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
        help="Nom du projet (défaut: {dataset}_hybrid)",
    )
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Problème de régression",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Nom de la configuration (default, fast, exhaustive, llmfe_only, dfs_only)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Nombre max de features (override)",
    )
    parser.add_argument(
        "--llmfe-iterations",
        type=int,
        default=None,
        help="Nombre d'itérations LLMFE (override)",
    )

    # Options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mode silencieux",
    )

    # API Key (optionnel - sinon utilise .env)
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Clé API OpenAI ou Anthropic (optionnel, sinon utilise .env)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="Fournisseur LLM (default: openai)",
    )

    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()

    # Lister les configs
    if args.list_configs:
        list_configs()
        return

    # Vérifier les arguments requis
    if not args.dataset or not args.target:
        print("Erreur: --dataset et --target sont requis")
        print("   Utilisez --list-configs pour voir les configurations disponibles")
        sys.exit(1)

    project_name = args.project or f"{args.dataset}_hybrid"

    # === Configurer l'API Key ===
    if args.api_key:
        # Clé passée en argument
        os.environ["API_KEY"] = args.api_key
        if args.provider == "openai":
            os.environ["OPENAI_API_KEY"] = args.api_key
        else:
            os.environ["ANTHROPIC_API_KEY"] = args.api_key
        print(f"[API] Clé {args.provider.upper()} configurée via --api-key")
    else:
        # Vérifier dans .env
        api_available, api_message = check_api_key(args.config)
        if api_available:
            print(f"[API] {api_message}")
        else:
            if args.config not in ["dfs_only"]:
                print(f"\n[WARNING] {api_message}")
                print("[WARNING] LLMFE sera désactivé. Utilisation de DFS seul.")
                print("[INFO] Pour activer LLMFE, ajoutez --api-key <votre_clé> ou configurez .env")
                # Forcer le mode DFS only
                args.config = "dfs_only"

    # Charger le dataset
    try:
        df_train = load_dataset(args.dataset, args.target)
    except FileNotFoundError as e:
        print(f"\nErreur: {e}")
        sys.exit(1)

    # Mode comparaison
    if args.compare_configs:
        compare_configs(
            df_train=df_train,
            target_col=args.target,
            project_name=project_name,
            is_regression=args.regression,
        )
        return

    # Mode normal : une seule configuration
    print("\n" + "=" * 70)
    print("  TEST FEATURE ENGINEERING HYBRIDE")
    print("=" * 70)
    print(f"  Dataset      : {args.dataset}")
    print(f"  Target       : {args.target}")
    print(f"  Project      : {project_name}")
    print(f"  Type         : {'Régression' if args.regression else 'Classification'}")
    print(f"  Config       : {args.config}")
    print("=" * 70)

    # Charger la configuration
    from src.feature_engineering.hybrid.config import get_hybrid_config

    config = get_hybrid_config(args.config)

    # Appliquer les overrides
    if args.max_features is not None:
        config.max_features = args.max_features
    if args.llmfe_iterations is not None:
        config.llmfe_max_iterations = args.llmfe_iterations

    config.verbose = not args.quiet

    # Exécuter
    engineer = HybridFeatureEngineer(project_name=project_name, config=config)
    result = engineer.run(
        df_train=df_train,
        target_col=args.target,
        is_regression=args.regression,
    )

    # Afficher les chemins de sortie
    print("\n" + "-" * 70)
    print("Fichiers générés:")
    print(f"   - outputs/{project_name}/feature_engineering/hybrid/train_hybrid.parquet")
    print(f"   - outputs/{project_name}/feature_engineering/hybrid/hybrid_report.json")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
