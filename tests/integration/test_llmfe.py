"""
Test d'intégration pour le module LLMFE (LLM-based Feature Engineering).
Génère automatiquement des features via LLM et algorithme évolutif.

Usage:
    python tests/integration/test_llmfe.py
    python tests/integration/test_llmfe.py --project titanic --max-samples 10
    python tests/integration/test_llmfe.py --project titanic --model gpt-4o-mini --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# Ajoute le dossier racine à sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Ajouter le dossier llmfe pour les imports internes du module LLMFE
# Le module est dans src/feature_engineering/llmfe/
llmfe_path = project_root / "src" / "feature_engineering" / "llmfe"
sys.path.insert(0, str(llmfe_path))

# Configuration environnement
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from src.core.config import settings
from src.core.io_utils import csv_to_dataframe_train_test
from src.feature_engineering.llmfe.feature_formatter import FeatureFormat
from src.feature_engineering.path_config import FeatureEngineeringPathConfig

# === Configuration des projets disponibles ===
PROJETS = {
    "titanic": {
        "label": "Survived",
        "path": "data/raw/titanic",
        "description": "Dataset Titanic - Prédiction de survie",
        "is_regression": False,
        "task_description": "Predict whether a passenger survived the Titanic disaster based on their characteristics.",
    },
    "verbatims": {
        "label": "Categorie",
        "path": "data/raw/verbatims",
        "description": "Classification de verbatims clients",
        "is_regression": False,
        "task_description": "Classify customer verbatims into predefined categories.",
    },
}


def run_llmfe_test(
    project_name: str = "titanic",
    max_samples: int = 10,
    model: str = "gpt-3.5-turbo",
    samples_per_prompt: int = 2,
    dry_run: bool = False,
    verbose: bool = True,
    feature_format: str = "basic",
    analyse_path: str = None,
) -> FeatureEngineeringPathConfig:
    """
    Exécute le test LLMFE sur un dataset.

    Args:
        project_name: Nom du projet (clé dans PROJETS)
        max_samples: Nombre maximum d'itérations LLM
        model: Modèle OpenAI à utiliser
        samples_per_prompt: Nombre de samples par appel API
        dry_run: Si True, prépare tout mais n'exécute pas LLMFE
        verbose: Affiche les informations de progression
        feature_format: Format des features (basic, tags, hierarchical)
        analyse_path: Chemin vers le JSON d'analyse existant (optionnel)

    Returns:
        FeatureEngineeringPathConfig avec les chemins des fichiers générés
    """
    if project_name not in PROJETS:
        available = ", ".join(PROJETS.keys())
        raise ValueError(f"Projet '{project_name}' inconnu. Disponibles: {available}")

    projet = PROJETS[project_name]

    print("\n" + "=" * 70)
    print(f"  LLMFE - {project_name.upper()}")
    print(f"  {projet['description']}")
    print("=" * 70)

    # === 1. Vérifier la clé API ===
    if not settings.is_configured("openai"):
        print("\n[ERROR] OPENAI_API_KEY non configurée!")
        print("Configurez votre clé dans le fichier .env")
        raise ValueError("OPENAI_API_KEY requise pour LLMFE")

    # === 2. Initialiser le gestionnaire de chemins ===
    path_config = FeatureEngineeringPathConfig(project_name=project_name)
    path_config.log(f"Démarrage LLMFE: {project_name}")

    if verbose:
        print(f"\n[INFO] Dossier de sortie: {path_config.project_dir}")

    # === 3. Charger les données ===
    print(f"\n[1/5] Chargement des données depuis {projet['path']}...")
    try:
        df_train, df_test = csv_to_dataframe_train_test(projet["path"])
        path_config.log(f"Données chargées: {len(df_train)} lignes train")
    except Exception as e:
        path_config.log(f"ERREUR chargement données: {e}", level="ERROR")
        raise

    target_col = projet["label"]
    feature_cols = [c for c in df_train.columns if c != target_col]

    print(f"    - Lignes train: {len(df_train)}")
    print(f"    - Lignes test: {len(df_test)}")
    print(f"    - Features: {len(feature_cols)}")
    print(f"    - Cible: {target_col}")

    # === 4. Préparer les métadonnées des features ===
    print("\n[2/5] Préparation des métadonnées...")

    # Générer des descriptions pour chaque feature
    meta_data = {}
    for col in feature_cols:
        # Description simple basée sur le nom de la colonne
        description = col.replace("_", " ").replace("-", " ")
        meta_data[col] = description

    if verbose:
        print(f"    Features avec métadonnées: {len(meta_data)}")

    # === 5. Configuration LLMFE ===
    print("\n[3/5] Configuration LLMFE...")
    print(f"    - Modèle: {model}")
    print(f"    - Max samples: {max_samples}")
    print(f"    - Samples/prompt: {samples_per_prompt}")
    print(f"    - Type: {'Régression' if projet['is_regression'] else 'Classification'}")

    # Sauvegarder la spec pour référence
    spec_content = _generate_spec(
        task_description=projet["task_description"],
        is_regression=projet["is_regression"],
    )
    path_config.save_spec(spec_content, name="specification")
    path_config.log("Spec générée et sauvegardée")

    # === 6. Exécution LLMFE ===
    if dry_run:
        print("\n[4/5] Mode dry-run - LLMFE non exécuté")
        print("    La configuration a été validée.")
        print("    Relancez sans --dry-run pour exécuter LLMFE.")
        path_config.log("Mode dry-run - LLMFE non exécuté")
    else:
        print("\n[4/5] Exécution de LLMFE...")
        path_config.log("Démarrage de l'exécution LLMFE")

        try:
            # Import du runner LLMFE depuis src/feature_engineering/llmfe/
            from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner

            # Configurer la clé API
            os.environ["API_KEY"] = settings.openai_api_key

            # Créer et exécuter le runner avec le path_config existant
            # (évite la création d'une structure dupliquée)
            runner = LLMFERunner(
                project_name=project_name,
                path_config=path_config,  # Réutiliser le path_config existant
            )

            # Convertir le format string en enum
            format_enum = FeatureFormat(feature_format.lower())

            result = runner.run(
                df_train=df_train,
                target_col=target_col,
                is_regression=projet["is_regression"],
                max_samples=max_samples,
                task_description=projet["task_description"],
                meta_data=meta_data,
                use_api=True,
                api_model=model,
                samples_per_prompt=samples_per_prompt,
                feature_format=format_enum,
                analyse_path=analyse_path,
            )

            path_config.log(f"LLMFE terminé - résultats dans {result.get('results_dir', 'N/A')}")

        except ImportError as e:
            print(f"\n[ERROR] Impossible d'importer LLMFE: {e}")
            print("Vérifiez que le module llmfe est correctement installé.")
            path_config.log(f"ERREUR import LLMFE: {e}", level="ERROR")
            raise
        except Exception as e:
            print(f"\n[ERROR] Erreur lors de l'exécution LLMFE: {e}")
            path_config.log(f"ERREUR LLMFE: {e}", level="ERROR")
            raise

    # === 7. Sauvegarder les métadonnées ===
    print("\n[5/5] Sauvegarde des métadonnées...")

    path_config.save_fe_metadata(
        dataset_name=project_name,
        target_col=target_col,
        n_rows_train=len(df_train),
        n_rows_test=len(df_test),
        n_original_features=len(feature_cols),
        n_final_features=len(feature_cols),  # Sera mis à jour après FE
        transforms_applied=["llmfe"],
        llmfe_used=not dry_run,
        llmfe_best_score=None,  # Sera mis à jour après exécution
    )

    # === Résumé final ===
    print("\n" + "=" * 70)
    print("  LLMFE TERMINE")
    print("=" * 70)
    print("\nFichiers générés:")
    for name, path in path_config.get_all_paths().items():
        if Path(path).exists():
            print(f"  - {name}: {path}")

    path_config.log("Test LLMFE terminé avec succès")

    return path_config


def _generate_spec(task_description: str, is_regression: bool) -> str:
    """Génère une spec dynamiquement selon le type de problème."""

    if is_regression:
        model = "xgb.XGBRegressor"
        metric_import = "from sklearn.metrics import mean_squared_error"
        score_calc = "score = -1 * mean_squared_error(y_test, y_pred, squared=False)"
        y_transform = "y = outputs"
        kfold = "kf = KFold(n_splits=4, shuffle=True, random_state=42)"
    else:
        model = "xgb.XGBClassifier"
        metric_import = "from sklearn.metrics import accuracy_score"
        score_calc = "score = accuracy_score(y_test, y_pred)"
        y_transform = "y = label_encoder.fit_transform(outputs)"
        kfold = "kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)"

    return f'''"""
[PREFIX]

###
<Task>
{task_description}

###
<Features>
[FEATURES]

###
<Examples>
[EXAMPLES]
[SUFFIX]
"""

@evaluate.run
def evaluate(data: dict):
    """Evaluate the feature transformations on data observations."""
    from sklearn import preprocessing
    from sklearn.model_selection import StratifiedKFold, KFold
    {metric_import}
    from preprocessing import preprocess_datasets
    import xgboost as xgb
    import numpy as np

    label_encoder = preprocessing.LabelEncoder()
    inputs, outputs, is_cat, is_regression = data['inputs'], data['outputs'], data['is_cat'], data['is_regression']
    X = modify_features(inputs)
    {y_transform}

    # Encode categorical string columns
    for col in X.columns:
        if X[col].dtype == 'string' or X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col].astype(str))

    {kfold}
    scores = []

    # 4-Fold Cross-Validation
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_new, X_test_new = preprocess_datasets(X_train, X_test, None)

        model = {model}(random_state=42)
        model.fit(X_train_new, y_train)
        y_pred = model.predict(X_test_new)
        {score_calc}
        scores.append(score)

    return np.mean(scores), inputs, outputs


@equation.evolve
def modify_features(df_input) -> pd.DataFrame:
    """
    Initial feature engineering function.
    This function will be evolved by the LLM to create better features.
    """
    import pandas as pd
    import numpy as np

    df_output = df_input.copy()
    return df_output
'''


def main():
    """Point d'entrée principal avec arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Exécute LLMFE (LLM-based Feature Engineering) sur un dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python tests/integration/test_llmfe.py
  python tests/integration/test_llmfe.py --project titanic --max-samples 10
  python tests/integration/test_llmfe.py --project titanic --model gpt-4o-mini
  python tests/integration/test_llmfe.py --dry-run  # Valide la config sans exécuter
        """,
    )

    parser.add_argument(
        "--project",
        "-p",
        choices=list(PROJETS.keys()),
        default="titanic",
        help="Projet à traiter (default: titanic)",
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=10,
        help="Nombre maximum d'itérations LLM (default: 10)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-4o-mini",
        help="Modèle OpenAI à utiliser (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=2,
        help="Nombre de samples générés par appel API (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Valide la configuration sans exécuter LLMFE"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Mode silencieux")
    parser.add_argument(
        "--feature-format",
        "-f",
        choices=["basic", "tags", "hierarchical"],
        default="basic",
        help="Format des features dans le prompt (default: basic)",
    )
    parser.add_argument(
        "--analyse-path",
        type=str,
        default=None,
        help="Chemin vers le JSON d'analyse existant (optionnel)",
    )

    args = parser.parse_args()

    try:
        run_llmfe_test(
            project_name=args.project,
            max_samples=args.max_samples,
            model=args.model,
            samples_per_prompt=args.samples_per_prompt,
            dry_run=args.dry_run,
            verbose=not args.quiet,
            feature_format=args.feature_format,
            analyse_path=args.analyse_path,
        )
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Erreur inattendue: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
