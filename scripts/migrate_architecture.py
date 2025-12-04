#!/usr/bin/env python3
"""
Script de migration de l'architecture du projet IA_create_IA_cate.

Ce script réorganise la structure des dossiers pour une meilleure clarté.
Il crée des copies (pas de suppression) pour permettre une migration progressive.

Usage:
    python scripts/migrate_architecture.py --dry-run    # Voir les changements sans les appliquer
    python scripts/migrate_architecture.py              # Appliquer les changements
    python scripts/migrate_architecture.py --cleanup    # Supprimer les anciens dossiers (après vérification)
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# MAPPING DES FICHIERS/DOSSIERS
# Format: (source, destination)
# ============================================================================

FOLDER_MIGRATIONS: list[tuple[str, str]] = [
    # Core - Fusion des utilitaires
    ("src/helper", "src/core"),
    ("src/fonctions", "src/core"),
    # Data loader
    ("src/Data", "src/data"),
    # Analyse - Réorganisation
    ("src/analyse/metier", "src/analyse/business"),
    ("src/analyse/statistiques", "src/analyse/stats"),
    ("src/analyse/dataset", "src/analyse/stats"),  # Fusion dans stats
    # Feature Engineering - Renommage et réorganisation
    ("src/features_engineering", "src/feature_engineering"),
    # AutoML - Renommage
    ("src/autoML_supervise", "src/automl/supervised"),
    ("src/autoML_nonsupervise", "src/automl/unsupervised"),
    # Tests - Renommage
    ("Test", "tests"),
    # Données - Réorganisation
    ("Data", "data/raw"),
    # Outputs - Renommage
    ("Creation", "outputs"),
    # Notebooks - Renommage
    ("Notebook", "notebooks"),
]

FILE_MIGRATIONS: list[tuple[str, str]] = [
    # Core - LLM Client unifié
    ("src/helper/ollama_llm.py", "src/core/llm_client.py"),
    ("src/helper/ddataframe.py", "src/core/dataframe_utils.py"),
    ("src/fonctions/csv.py", "src/core/io_utils.py"),
    ("src/fonctions/format_entrainement.py", "src/core/preprocessing.py"),
    ("src/fonctions/clean_label.py", "src/core/text_cleaning.py"),
    ("src/fonctions/clean_tokeniser.py", "src/core/text_cleaning.py"),  # À fusionner
    # Data loader
    ("src/Data/load_datasets.py", "src/data/loader.py"),
    # Analyse
    ("src/analyse/analyse.py", "src/analyse/analyser.py"),
    # Feature Engineering - Declarative
    (
        "src/features_engineering/LLM/analyse_fe/pipeline.py",
        "src/feature_engineering/declarative/planner.py",
    ),
    (
        "src/features_engineering/LLM/transcriptions_fe/pipeline.py",
        "src/feature_engineering/declarative/transformer.py",
    ),
    (
        "src/features_engineering/LLM/code_fe/pipeline.py",
        "src/feature_engineering/declarative/code_generator.py",
    ),
    # Feature Engineering - Transforms
    (
        "src/features_engineering/transformation_fe/numeric_transforms.py",
        "src/feature_engineering/transforms/numeric.py",
    ),
    (
        "src/features_engineering/transformation_fe/categorical_transforms.py",
        "src/feature_engineering/transforms/categorical.py",
    ),
    (
        "src/features_engineering/transformation_fe/datetime_transforms.py",
        "src/feature_engineering/transforms/datetime.py",
    ),
    (
        "src/features_engineering/transformation_fe/text_transforms.py",
        "src/feature_engineering/transforms/text.py",
    ),
    (
        "src/features_engineering/transformation_fe/registry.py",
        "src/feature_engineering/transforms/registry.py",
    ),
    (
        "src/features_engineering/transformation_fe/apply_plan.py",
        "src/feature_engineering/transforms/apply_plan.py",
    ),
    # Feature Engineering - Libs
    (
        "src/features_engineering/lib_existante/feature_engine.py",
        "src/feature_engineering/libs/feature_engine_wrapper.py",
    ),
    (
        "src/features_engineering/lib_existante/feature_tools.py",
        "src/feature_engineering/libs/featuretools_wrapper.py",
    ),
    # AutoML
    ("src/autoML_supervise/all_autoML.py", "src/automl/runner.py"),
    ("src/autoML_supervise/flaml.py", "src/automl/supervised/flaml_wrapper.py"),
    ("src/autoML_supervise/autogluon.py", "src/automl/supervised/autogluon_wrapper.py"),
    ("src/autoML_supervise/tpot1.py", "src/automl/supervised/tpot_wrapper.py"),
    ("src/autoML_supervise/h2o/h2o.py", "src/automl/supervised/h2o_wrapper.py"),
    # Pipeline
    ("src/pipeline/pipeline_autoMl.py", "src/pipeline/full_pipeline.py"),
    # Tests
    ("Test/AutoML/test_all.py", "tests/integration/test_automl.py"),
    ("Test/AutoML/test_flaml.py", "tests/unit/test_flaml.py"),
    ("Test/AutoML/test_autogluon.py", "tests/unit/test_autogluon.py"),
    ("Test/AutoML/test_tpot.py", "tests/unit/test_tpot.py"),
    ("Test/AutoML/test_h2o.py", "tests/unit/test_h2o.py"),
    ("Test/analyse/test_analyse.py", "tests/unit/test_analyse.py"),
    ("Test/feature_engineering/test_fe.py", "tests/unit/test_feature_engineering.py"),
    ("Test/pipeline/test_pipeline.py", "tests/integration/test_pipeline.py"),
]

# Nouveaux dossiers à créer
NEW_DIRECTORIES: list[str] = [
    "src/core",
    "src/data",
    "src/analyse/stats",
    "src/analyse/business",
    "src/feature_engineering",
    "src/feature_engineering/llmfe",
    "src/feature_engineering/declarative",
    "src/feature_engineering/transforms",
    "src/feature_engineering/libs",
    "src/automl",
    "src/automl/supervised",
    "src/automl/unsupervised",
    "tests",
    "tests/unit",
    "tests/integration",
    "data/raw",
    "data/processed",
    "outputs",
    "notebooks",
    "config",
    "config/prompts",
    "docs",
    "scripts",
]

# Fichiers __init__.py à créer
INIT_FILES: list[str] = [
    "src/__init__.py",
    "src/core/__init__.py",
    "src/data/__init__.py",
    "src/analyse/__init__.py",
    "src/analyse/stats/__init__.py",
    "src/analyse/business/__init__.py",
    "src/feature_engineering/__init__.py",
    "src/feature_engineering/llmfe/__init__.py",
    "src/feature_engineering/declarative/__init__.py",
    "src/feature_engineering/transforms/__init__.py",
    "src/feature_engineering/libs/__init__.py",
    "src/automl/__init__.py",
    "src/automl/supervised/__init__.py",
    "src/automl/unsupervised/__init__.py",
    "src/pipeline/__init__.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
]


class ArchitectureMigrator:
    """Gère la migration de l'architecture du projet."""

    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.log: list[str] = []

    def _log(self, message: str, level: str = "INFO"):
        """Ajoute un message au log."""
        prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "SKIP": "⏭️"}
        formatted = f"{prefix.get(level, '•')} {message}"
        self.log.append(formatted)
        print(formatted)

    def _resolve_path(self, relative_path: str) -> Path:
        """Résout un chemin relatif par rapport à la racine du projet."""
        return self.project_root / relative_path

    def create_directories(self):
        """Crée les nouveaux dossiers."""
        self._log("=" * 60)
        self._log("CRÉATION DES NOUVEAUX DOSSIERS")
        self._log("=" * 60)

        for dir_path in NEW_DIRECTORIES:
            full_path = self._resolve_path(dir_path)
            if full_path.exists():
                self._log(f"Dossier existe déjà: {dir_path}", "SKIP")
            else:
                if not self.dry_run:
                    full_path.mkdir(parents=True, exist_ok=True)
                self._log(f"Créé: {dir_path}", "SUCCESS")

    def create_init_files(self):
        """Crée les fichiers __init__.py."""
        self._log("=" * 60)
        self._log("CRÉATION DES FICHIERS __init__.py")
        self._log("=" * 60)

        for init_path in INIT_FILES:
            full_path = self._resolve_path(init_path)
            if full_path.exists():
                self._log(f"Existe déjà: {init_path}", "SKIP")
            else:
                if not self.dry_run:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(
                        f'# {full_path.parent.name}\n"""Module {full_path.parent.name}."""\n'
                    )
                self._log(f"Créé: {init_path}", "SUCCESS")

    def migrate_files(self):
        """Migre les fichiers individuels."""
        self._log("=" * 60)
        self._log("MIGRATION DES FICHIERS")
        self._log("=" * 60)

        for source, dest in FILE_MIGRATIONS:
            source_path = self._resolve_path(source)
            dest_path = self._resolve_path(dest)

            if not source_path.exists():
                self._log(f"Source inexistante: {source}", "WARNING")
                continue

            if dest_path.exists():
                self._log(f"Destination existe déjà: {dest}", "SKIP")
                continue

            if not self.dry_run:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)

            self._log(f"{source} → {dest}", "SUCCESS")

    def migrate_llmfe(self):
        """Migre le module LLMFE."""
        self._log("=" * 60)
        self._log("MIGRATION DE LLMFE")
        self._log("=" * 60)

        source = self._resolve_path("LLMFE/llmfe")
        dest = self._resolve_path("src/feature_engineering/llmfe")

        if not source.exists():
            self._log("LLMFE/llmfe non trouvé", "WARNING")
            return

        if dest.exists() and any(dest.iterdir()):
            self._log("src/feature_engineering/llmfe existe déjà", "SKIP")
            return

        if not self.dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                if item.name == "__pycache__":
                    continue
                dest_item = dest / item.name
                if item.is_file():
                    shutil.copy2(item, dest_item)
                else:
                    shutil.copytree(item, dest_item)

        self._log("LLMFE/llmfe → src/feature_engineering/llmfe", "SUCCESS")

        # Copier aussi les prompts
        prompts_source = self._resolve_path("LLMFE/prompts")
        prompts_dest = self._resolve_path("config/prompts")

        if prompts_source.exists() and not any(prompts_dest.glob("*.txt")):
            if not self.dry_run:
                for item in prompts_source.glob("*.txt"):
                    shutil.copy2(item, prompts_dest / item.name)
            self._log("LLMFE/prompts → config/prompts", "SUCCESS")

    def migrate_data(self):
        """Migre les données."""
        self._log("=" * 60)
        self._log("MIGRATION DES DONNÉES")
        self._log("=" * 60)

        source = self._resolve_path("Data")
        dest = self._resolve_path("data/raw")

        if not source.exists():
            self._log("Data/ non trouvé", "WARNING")
            return

        for item in source.iterdir():
            if item.name.startswith("."):
                continue
            dest_item = dest / item.name
            if dest_item.exists():
                self._log(f"Existe déjà: data/raw/{item.name}", "SKIP")
                continue
            if not self.dry_run:
                if item.is_dir():
                    shutil.copytree(item, dest_item)
                else:
                    shutil.copy2(item, dest_item)
            self._log(f"Data/{item.name} → data/raw/{item.name}", "SUCCESS")

    def create_conftest(self):
        """Crée le fichier conftest.py pour pytest."""
        self._log("=" * 60)
        self._log("CRÉATION DE conftest.py")
        self._log("=" * 60)

        conftest_path = self._resolve_path("tests/conftest.py")

        if conftest_path.exists():
            self._log("tests/conftest.py existe déjà", "SKIP")
            return

        conftest_content = '''"""
Fixtures pytest partagées pour les tests.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Ajouter src au PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_dataframe():
    """DataFrame d'exemple pour les tests."""
    return pd.DataFrame({
        "feature_num": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_cat": ["A", "B", "A", "B", "A"],
        "target": [0, 1, 0, 1, 0],
    })


@pytest.fixture
def project_root():
    """Racine du projet."""
    return PROJECT_ROOT


@pytest.fixture
def data_dir(project_root):
    """Dossier des données."""
    return project_root / "data" / "raw"


@pytest.fixture
def outputs_dir(project_root):
    """Dossier des sorties."""
    return project_root / "outputs"
'''

        if not self.dry_run:
            conftest_path.write_text(conftest_content)
        self._log("Créé: tests/conftest.py", "SUCCESS")

    def generate_import_mapping(self) -> dict[str, str]:
        """Génère le mapping des anciens imports vers les nouveaux."""
        return {
            # Core
            "src.helper.ollama_llm": "src.core.llm_client",
            "src.fonctions.csv": "src.core.io_utils",
            "src.fonctions.format_entrainement": "src.core.preprocessing",
            "src.fonctions.clean_label": "src.core.text_cleaning",
            "src.helper.ddataframe": "src.core.dataframe_utils",
            # Data
            "src.Data.load_datasets": "src.data.loader",
            # Analyse
            "src.analyse.metier": "src.analyse.business",
            "src.analyse.statistiques": "src.analyse.stats",
            # Feature Engineering
            "src.features_engineering": "src.feature_engineering",
            "src.features_engineering.LLM.analyse_fe": "src.feature_engineering.declarative",
            "src.features_engineering.LLM.transcriptions_fe": "src.feature_engineering.declarative",
            "src.features_engineering.LLM.code_fe": "src.feature_engineering.declarative",
            "src.features_engineering.transformation_fe": "src.feature_engineering.transforms",
            "src.features_engineering.lib_existante": "src.feature_engineering.libs",
            # AutoML
            "src.autoML_supervise": "src.automl.supervised",
            "src.autoML_nonsupervise": "src.automl.unsupervised",
            "src.autoML_supervise.all_autoML": "src.automl.runner",
        }

    def save_import_mapping(self):
        """Sauvegarde le mapping des imports pour référence."""
        self._log("=" * 60)
        self._log("GÉNÉRATION DU MAPPING DES IMPORTS")
        self._log("=" * 60)

        mapping = self.generate_import_mapping()
        mapping_path = self._resolve_path("docs/import_migration.md")

        content = """# Guide de migration des imports

Ce fichier liste les anciens imports et leurs équivalents dans la nouvelle architecture.

## Mapping

| Ancien import | Nouvel import |
|---------------|---------------|
"""
        for old, new in sorted(mapping.items()):
            content += f"| `{old}` | `{new}` |\n"

        content += """
## Exemple de migration

```python
# Avant
from src.fonctions.csv import to_csv
from src.autoML_supervise.all_autoML import all_autoML
from src.features_engineering.LLM.analyse_fe.pipeline import LLMFeatureEngineeringPipeline

# Après
from src.core.io_utils import to_csv
from src.automl.runner import all_autoML
from src.feature_engineering.declarative.planner import LLMFeatureEngineeringPipeline
```

## Script de remplacement automatique

Pour mettre à jour vos imports automatiquement, vous pouvez utiliser sed:

```bash
# Exemple pour un fichier
sed -i '' 's/src.fonctions.csv/src.core.io_utils/g' votre_fichier.py
```
"""

        if not self.dry_run:
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_path.write_text(content)
        self._log("Créé: docs/import_migration.md", "SUCCESS")

    def run(self):
        """Exécute la migration complète."""
        start_time = datetime.now()

        print("\n" + "=" * 60)
        print("   MIGRATION DE L'ARCHITECTURE DU PROJET")
        print("=" * 60)
        print(f"   Mode: {'DRY RUN (simulation)' if self.dry_run else 'RÉEL'}")
        print(f"   Racine: {self.project_root}")
        print("=" * 60 + "\n")

        # Exécuter les étapes
        self.create_directories()
        print()
        self.create_init_files()
        print()
        self.migrate_files()
        print()
        self.migrate_llmfe()
        print()
        self.migrate_data()
        print()
        self.create_conftest()
        print()
        self.save_import_mapping()

        # Résumé
        duration = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("   MIGRATION TERMINÉE")
        print("=" * 60)
        print(f"   Durée: {duration.total_seconds():.2f}s")
        print(f"   Mode: {'DRY RUN (aucun changement appliqué)' if self.dry_run else 'RÉEL'}")
        print("=" * 60)

        if self.dry_run:
            print("\n💡 Pour appliquer les changements, relancez sans --dry-run")
        else:
            print("\n✅ Migration terminée!")
            print("📝 Consultez docs/import_migration.md pour mettre à jour vos imports")
            print("⚠️  Les anciens dossiers n'ont PAS été supprimés (migration progressive)")


def main():
    parser = argparse.ArgumentParser(description="Migration de l'architecture du projet")
    parser.add_argument(
        "--dry-run", action="store_true", help="Simuler sans appliquer les changements"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Supprimer les anciens dossiers (à utiliser après vérification)",
    )
    args = parser.parse_args()

    if args.cleanup:
        print("⚠️  La fonctionnalité --cleanup n'est pas encore implémentée.")
        print(
            "    Supprimez manuellement les anciens dossiers après avoir vérifié que tout fonctionne."
        )
        return

    migrator = ArchitectureMigrator(PROJECT_ROOT, dry_run=args.dry_run)
    migrator.run()


if __name__ == "__main__":
    main()
