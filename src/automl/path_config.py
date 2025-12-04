# src/automl/path_config.py
"""
Gestionnaire de chemins pour le module AutoML.
Hérite de BasePathConfig pour la gestion centralisée.

Structure des dossiers créés:
{output_dir}/
└── {project_name}/
    └── automl/
        ├── flaml/
        │   └── time_budget_{N}/
        ├── autogluon/
        │   └── time_budget_{N}/
        ├── tpot/
        │   └── time_budget_{N}/
        ├── h2o/
        │   └── time_budget_{N}/
        ├── results/
        │   └── comparison.json
        └── logs/
            └── automl.log

Note: Le timestamp n'est plus dans la structure des dossiers,
il est stocké uniquement dans les métadonnées.
"""

from __future__ import annotations

from pathlib import Path

from src.core.base_path_config import BasePathConfig


class AutoMLPathConfig(BasePathConfig):
    """
    Configuration des chemins pour les sorties AutoML.

    Hérite de BasePathConfig et ajoute les sous-dossiers spécifiques:
    - flaml/      : Modèles FLAML
    - autogluon/  : Modèles AutoGluon
    - tpot/       : Modèles TPOT
    - h2o/        : Modèles H2O
    - results/    : Comparaisons et résumés
    """

    MODULE_NAME = "automl"

    # Frameworks supportés
    FRAMEWORKS = ["flaml", "autogluon", "tpot", "h2o"]

    def __init__(
        self,
        project_name: str,
        base_dir: str | Path | None = None,
        time_budget: int = 60,
    ):
        """
        Initialise la configuration des chemins pour AutoML.

        Args:
            project_name: Nom du projet (ex: "titanic", "verbatims")
            base_dir: Dossier racine optionnel (sinon lu depuis Settings.output_dir)
            time_budget: Budget temps en secondes pour l'entraînement
        """
        self.time_budget = time_budget

        # Appel du constructeur parent
        super().__init__(
            project_name=project_name,
            base_dir=base_dir,
        )

        # Sous-dossiers par framework
        self.flaml_dir = self.project_dir / "flaml" / f"time_budget_{time_budget}"
        self.autogluon_dir = self.project_dir / "autogluon" / f"time_budget_{time_budget}"
        self.tpot_dir = self.project_dir / "tpot" / f"time_budget_{time_budget}"
        self.h2o_dir = self.project_dir / "h2o" / f"time_budget_{time_budget}"

        # Dossier des résultats comparatifs
        self.results_dir = self.project_dir / "results"

        # Créer les dossiers
        self._create_directories()

    # === Implémentation des méthodes abstraites ===

    def _get_subdirectories(self) -> list[Path]:
        """Retourne les sous-dossiers spécifiques à AutoML."""
        return [
            self.flaml_dir,
            self.autogluon_dir,
            self.tpot_dir,
            self.h2o_dir,
            self.results_dir,
        ]

    def get_all_paths(self) -> dict[str, str]:
        """Retourne tous les chemins configurés."""
        paths = self.get_base_paths()
        paths.update(
            {
                "flaml_dir": str(self.flaml_dir),
                "autogluon_dir": str(self.autogluon_dir),
                "tpot_dir": str(self.tpot_dir),
                "h2o_dir": str(self.h2o_dir),
                "results_dir": str(self.results_dir),
                "comparison_path": str(self.comparison_path),
                "leaderboard_path": str(self.leaderboard_path),
            }
        )
        return paths

    # === Chemins des fichiers spécifiques ===

    @property
    def comparison_path(self) -> Path:
        """Chemin du fichier de comparaison des frameworks."""
        return self.results_dir / "comparison.json"

    @property
    def leaderboard_path(self) -> Path:
        """Chemin du leaderboard global."""
        return self.results_dir / "leaderboard.csv"

    def get_framework_dir(self, framework: str) -> Path:
        """
        Retourne le dossier d'un framework spécifique.

        Args:
            framework: Nom du framework ("flaml", "autogluon", "tpot", "h2o")

        Returns:
            Path du dossier du framework
        """
        framework = framework.lower()
        mapping = {
            "flaml": self.flaml_dir,
            "autogluon": self.autogluon_dir,
            "tpot": self.tpot_dir,
            "h2o": self.h2o_dir,
        }
        if framework not in mapping:
            raise ValueError(
                f"Framework '{framework}' inconnu. Disponibles: {list(mapping.keys())}"
            )
        return mapping[framework]

    def get_model_path(self, framework: str, filename: str = "model.pkl") -> Path:
        """
        Retourne le chemin du modèle pour un framework.

        Args:
            framework: Nom du framework
            filename: Nom du fichier modèle

        Returns:
            Path du fichier modèle
        """
        return self.get_framework_dir(framework) / filename

    def get_predictions_path(self, framework: str) -> Path:
        """
        Retourne le chemin des prédictions pour un framework.

        Args:
            framework: Nom du framework

        Returns:
            Path du fichier de prédictions
        """
        return self.get_framework_dir(framework) / "predictions_test.csv"

    # === Méthodes de sauvegarde spécifiques ===

    def save_comparison(self, scores: dict[str, float]) -> Path:
        """
        Sauvegarde la comparaison des scores entre frameworks.

        Args:
            scores: Dictionnaire {framework: score}

        Returns:
            Chemin du fichier sauvegardé
        """
        comparison = {
            "time_budget": self.time_budget,
            "scores": scores,
            "best_framework": max(scores, key=scores.get) if scores else None,
            "best_score": max(scores.values()) if scores else None,
        }
        self.log(f"Sauvegarde comparaison: {scores}")
        return self.save_json(comparison, self.comparison_path)

    def save_automl_metadata(
        self,
        dataset_name: str,
        target_col: str,
        n_rows: int,
        n_features: int,
        frameworks_run: list[str],
        scores: dict[str, float],
    ) -> Path:
        """
        Sauvegarde les métadonnées spécifiques à AutoML.

        Args:
            dataset_name: Nom du dataset
            target_col: Colonne cible
            n_rows: Nombre de lignes
            n_features: Nombre de features
            frameworks_run: Liste des frameworks exécutés
            scores: Scores par framework

        Returns:
            Chemin du fichier de métadonnées
        """
        metadata = {
            "dataset_name": dataset_name,
            "target_column": target_col,
            "n_rows": n_rows,
            "n_features": n_features,
            "time_budget": self.time_budget,
            "frameworks_run": frameworks_run,
            "scores": scores,
            "best_framework": max(scores, key=scores.get) if scores else None,
        }
        return self.save_metadata(metadata)

    # === Méthode pour obtenir le chemin legacy (compatibilité) ===

    def get_legacy_path(self) -> str:
        """
        Retourne le chemin au format legacy (Modeles/{projet}).

        Pour compatibilité avec l'ancien code qui utilise self.Nom_dossier.

        Returns:
            Chemin au format string
        """
        return str(self.project_dir)
