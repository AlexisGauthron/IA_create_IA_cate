"""
Test d'intégration pour le module d'analyse.
Génère l'analyse complète d'un dataset avec gestion des chemins.

Usage:
    python tests/integration/test_analyse.py --dataset titanic --target Survived
    python tests/integration/test_analyse.py --dataset verbatims --target Categorie --with-llm
    python tests/integration/test_analyse.py --dataset titanic --target Survived --provider openai --model gpt-4o
"""

import argparse
import os
import sys
from pathlib import Path

# Ajoute le dossier racine à sys.path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Configuration environnement
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd

import src.analyse.statistiques.report as report
from src.analyse.path_config import AnalysePathConfig
from src.core.config import settings

# =============================================================================
# Fonctions utilitaires
# =============================================================================


def detect_separator(file_path: Path) -> str:
    """
    Détecte automatiquement le séparateur d'un fichier CSV.

    Analyse la première ligne et compte les occurrences des séparateurs courants.
    """
    with open(file_path, encoding="utf-8") as f:
        first_line = f.readline()

    # Candidats séparateurs
    candidates = {",": 0, ";": 0, "\t": 0, "|": 0}
    for sep in candidates:
        candidates[sep] = first_line.count(sep)

    # Retourner celui qui apparaît le plus
    best_sep = max(candidates, key=candidates.get)

    # Si aucun séparateur trouvé, défaut à virgule
    if candidates[best_sep] == 0:
        return ","

    return best_sep


def load_data(dataset_name: str) -> pd.DataFrame:
    """
    Charge un dataset depuis data/raw/{dataset_name}/train.csv.

    Auto-détecte le séparateur CSV.
    Si le dataset n'existe pas, liste les datasets disponibles.
    """
    data_raw_dir = ROOT_DIR / "data" / "raw"
    dataset_dir = data_raw_dir / dataset_name

    # Vérifier si le dossier existe
    if not dataset_dir.exists():
        available_datasets = [
            d.name for d in data_raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' non trouvé dans data/raw/.\n"
            f"Datasets disponibles: {available_datasets}"
        )

    # Chercher train.csv
    data_path = dataset_dir / "train.csv"

    if not data_path.exists():
        available_files = [f.name for f in dataset_dir.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"Fichier 'train.csv' non trouvé dans data/raw/{dataset_name}/.\n"
            f"Fichiers disponibles: {available_files}"
        )

    # Auto-détecter le séparateur
    sep = detect_separator(data_path)
    print(f"[INFO] Séparateur détecté: '{sep}'")

    # Charger le CSV
    df = pd.read_csv(data_path, sep=sep, encoding="utf-8")
    print(f"[INFO] Dataset '{dataset_name}' chargé: {len(df)} lignes, {len(df.columns)} colonnes")

    return df


# =============================================================================
# Fonction principale d'analyse
# =============================================================================


def run_analyse(
    dataset_name: str,
    target_col: str,
    project_name: str = None,
    only_stats: bool = True,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> AnalysePathConfig:
    """
    Exécute l'analyse complète d'un dataset.

    Args:
        dataset_name: Nom du dossier dans data/raw/
        target_col: Colonne cible à prédire
        project_name: Nom du projet (défaut: dataset_name)
        only_stats: Si True, génère uniquement les stats (sans LLM)
        provider: Provider LLM ("openai", "ollama")
        model: Modèle LLM à utiliser
        verbose: Affiche les informations de progression

    Returns:
        AnalysePathConfig avec les chemins des fichiers générés
    """
    # Nom du projet = nom du dataset si non spécifié
    if project_name is None:
        project_name = dataset_name

    print("\n" + "=" * 70)
    print(f"  ANALYSE - {project_name.upper()}")
    print(f"  Dataset: {dataset_name} | Target: {target_col}")
    print("=" * 70)

    # === 1. Initialiser le gestionnaire de chemins ===
    path_config = AnalysePathConfig(project_name=project_name)
    path_config.log(f"Démarrage analyse: {project_name}")

    if verbose:
        print(f"\n[INFO] Dossier de sortie: {path_config.project_dir}")

    # === 2. Charger les données ===
    print(f"\n[1/4] Chargement des données depuis data/raw/{dataset_name}...")
    try:
        df_train = load_data(dataset_name)
        path_config.log(f"Données chargées: {len(df_train)} lignes train")
    except Exception as e:
        path_config.log(f"ERREUR chargement données: {e}")
        raise

    # Vérifier que la colonne cible existe
    if target_col not in df_train.columns:
        available_cols = list(df_train.columns)
        raise ValueError(
            f"Colonne cible '{target_col}' non trouvée.\n"
            f"Colonnes disponibles ({len(available_cols)}): {available_cols}"
        )

    feature_cols = [c for c in df_train.columns if c != target_col]

    print(f"    - Lignes: {len(df_train)}")
    print(f"    - Features: {len(feature_cols)}")
    print(f"    - Cible: {target_col}")

    # === 3. Générer l'analyse statistique ===
    print("\n[2/4] Génération du rapport statistique...")

    try:
        report_data = report.analyze_dataset_for_fe(
            df_train,
            target_cols=target_col,
            print_report=verbose,
            dataset_name=project_name,
            business_description=f"Analyse du dataset {dataset_name}",
        )
        path_config.log("Rapport statistique généré")
    except Exception as e:
        path_config.log(f"ERREUR analyse: {e}")
        raise

    # === 4. Sauvegarder le rapport stats ===
    print("\n[3/4] Sauvegarde du rapport statistique...")

    stats_payload = report_data.get("llm_payload", report_data)
    path_config.save_stats_report(stats_payload)

    # === 5. Analyse LLM (optionnelle) ===
    if not only_stats:
        print(f"\n[4/4] Analyse LLM avec {provider}/{model}...")

        # Vérifier la configuration API
        if provider == "openai" and not settings.is_configured("openai"):
            print("[WARNING] OPENAI_API_KEY non configurée. Passage en mode stats only.")
            path_config.log("OPENAI_API_KEY manquante - mode stats only")
            only_stats = True
        else:
            try:
                from src.analyse.metier.business_agent import run_business_clarification

                # Lancer l'agent de clarification métier
                full_payload = run_business_clarification(
                    stats_payload=stats_payload,
                    provider=provider,
                    model=model,
                    interactive=True,
                    verbose=verbose,
                )

                # Sauvegarder le rapport enrichi
                path_config.save_full_report(full_payload)
                path_config.log("Rapport complet avec LLM sauvegardé")

            except Exception as e:
                print(f"[ERROR] Erreur lors de l'analyse LLM: {e}")
                path_config.log(f"ERREUR LLM: {e}")
                import traceback

                traceback.print_exc()
                only_stats = True
    else:
        print("\n[4/4] Mode stats only - pas d'analyse LLM")

    # === 6. Sauvegarder les métadonnées ===
    path_config.save_analyse_metadata(
        dataset_name=dataset_name,
        target_col=target_col,
        n_rows=len(df_train),
        n_features=len(feature_cols),
        provider=provider if not only_stats else None,
        model=model if not only_stats else None,
        only_stats=only_stats,
    )

    # === Résumé final ===
    print("\n" + "=" * 70)
    print("  ANALYSE TERMINEE")
    print("=" * 70)
    print("\nFichiers générés:")
    for name, path in path_config.get_all_paths().items():
        if Path(path).exists():
            print(f"  - {name}: {path}")

    path_config.log("Analyse terminée avec succès")

    return path_config


# =============================================================================
# CLI
# =============================================================================


def main():
    """Point d'entrée principal avec arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Génère l'analyse d'un dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python tests/integration/test_analyse.py --dataset titanic --target Survived
  python tests/integration/test_analyse.py --dataset verbatims --target Categorie
  python tests/integration/test_analyse.py --dataset titanic --target Survived --with-llm --model gpt-4o

Le séparateur CSV est auto-détecté (virgule, point-virgule, tabulation, pipe).
        """,
    )

    # Paramètres requis
    parser.add_argument(
        "--dataset", "-d", type=str, required=True, help="[REQUIS] Nom du dataset dans data/raw/"
    )
    parser.add_argument(
        "--target", "-t", type=str, required=True, help="[REQUIS] Colonne cible à prédire"
    )

    # Paramètres optionnels
    parser.add_argument(
        "--project", "-p", type=str, default=None, help="Nom du projet (défaut: nom du dataset)"
    )
    parser.add_argument("--with-llm", action="store_true", help="Activer l'analyse LLM interactive")
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="openai",
        help="Provider LLM (défaut: openai)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Modèle LLM (défaut: gpt-4o-mini)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Mode silencieux")

    args = parser.parse_args()

    # Exécuter l'analyse
    run_analyse(
        dataset_name=args.dataset,
        target_col=args.target,
        project_name=args.project,
        only_stats=not args.with_llm,
        provider=args.provider,
        model=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
