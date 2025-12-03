# src/feature_engineering/path_config.py
"""
Gestionnaire de chemins pour le module Feature Engineering.
Hérite de BasePathConfig pour la gestion centralisée.

Structure des dossiers créés:
{output_dir}/
└── {project_name}/
    └── feature_engineering/
        ├── features/              # Features générées
        │   ├── train_fe.parquet
        │   └── test_fe.parquet
        ├── llmfe/                 # LLMFE spécifique
        │   ├── samples/
        │   ├── tensorboard/
        │   └── results/
        ├── transforms/            # Transformations appliquées
        │   └── pipeline.pkl
        ├── specs/                 # Spécifications
        └── logs/
            └── feature_engineering.log

Note: Le timestamp n'est plus dans la structure des dossiers,
il est stocké uniquement dans les métadonnées.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from src.core.base_path_config import BasePathConfig


class FeatureEngineeringPathConfig(BasePathConfig):
    """
    Configuration des chemins pour les sorties de Feature Engineering.

    Hérite de BasePathConfig et ajoute les sous-dossiers spécifiques:
    - features/    : DataFrames avec features générées
    - llmfe/       : Outputs du module LLMFE
    - transforms/  : Pipelines de transformation sauvegardés
    - specs/       : Spécifications et configurations
    """

    MODULE_NAME = "feature_engineering"

    def __init__(
        self,
        project_name: str,
        base_dir: Optional[str | Path] = None,
    ):
        """
        Initialise la configuration des chemins pour Feature Engineering.

        Args:
            project_name: Nom du projet (ex: "titanic", "verbatims")
            base_dir: Dossier racine optionnel (sinon lu depuis Settings.output_dir)
        """
        # Appel du constructeur parent
        super().__init__(
            project_name=project_name,
            base_dir=base_dir,
        )

        # Sous-dossiers spécifiques au Feature Engineering
        self.features_dir = self.project_dir / "features"
        self.llmfe_dir = self.project_dir / "llmfe"
        self.transforms_dir = self.project_dir / "transforms"
        self.specs_dir = self.project_dir / "specs"
        self.dataset_fe_dir = self.project_dir / "dataset_fe"  # Dataset transformé CSV

        # Sous-dossiers LLMFE
        self.llmfe_samples_dir = self.llmfe_dir / "samples"
        self.llmfe_tensorboard_dir = self.llmfe_dir / "tensorboard"
        self.llmfe_results_dir = self.llmfe_dir / "results"

        # Créer les dossiers
        self._create_directories()

    # === Implémentation des méthodes abstraites ===

    def _get_subdirectories(self) -> List[Path]:
        """Retourne les sous-dossiers spécifiques au Feature Engineering."""
        return [
            self.features_dir,
            self.llmfe_dir,
            self.llmfe_samples_dir,
            self.llmfe_tensorboard_dir,
            self.llmfe_results_dir,
            self.transforms_dir,
            self.specs_dir,
            self.dataset_fe_dir,
        ]

    def get_all_paths(self) -> Dict[str, str]:
        """Retourne tous les chemins configurés."""
        paths = self.get_base_paths()
        paths.update({
            "features_dir": str(self.features_dir),
            "llmfe_dir": str(self.llmfe_dir),
            "llmfe_samples_dir": str(self.llmfe_samples_dir),
            "llmfe_tensorboard_dir": str(self.llmfe_tensorboard_dir),
            "llmfe_results_dir": str(self.llmfe_results_dir),
            "transforms_dir": str(self.transforms_dir),
            "specs_dir": str(self.specs_dir),
            "train_features_path": str(self.train_features_path),
            "test_features_path": str(self.test_features_path),
        })
        return paths

    # === Chemins des fichiers de features ===

    @property
    def train_features_path(self) -> Path:
        """Chemin du fichier train avec features."""
        return self.features_dir / "train_fe.parquet"

    @property
    def test_features_path(self) -> Path:
        """Chemin du fichier test avec features."""
        return self.features_dir / "test_fe.parquet"

    @property
    def feature_pipeline_path(self) -> Path:
        """Chemin du pipeline de transformation sauvegardé."""
        return self.transforms_dir / "pipeline.pkl"

    @property
    def feature_columns_path(self) -> Path:
        """Chemin du fichier listant les features générées."""
        return self.features_dir / "feature_columns.json"

    # === Chemins dataset_fe (CSV transformé) ===

    @property
    def train_fe_csv_path(self) -> Path:
        """Chemin du fichier train transformé en CSV."""
        return self.dataset_fe_dir / "train_fe.csv"

    @property
    def test_fe_csv_path(self) -> Path:
        """Chemin du fichier test transformé en CSV."""
        return self.dataset_fe_dir / "test_fe.csv"

    # === Chemins LLMFE ===

    def get_llmfe_sample_path(self, sample_order: int) -> Path:
        """Retourne le chemin d'un fichier sample LLMFE."""
        return self.llmfe_samples_dir / f"sample_{sample_order:04d}.json"

    @property
    def llmfe_best_model_path(self) -> Path:
        """Chemin du meilleur modèle LLMFE."""
        return self.llmfe_results_dir / "best_model.json"

    @property
    def llmfe_all_scores_path(self) -> Path:
        """Chemin de tous les scores LLMFE."""
        return self.llmfe_results_dir / "all_scores.json"

    # === Chemins des specs ===

    def get_spec_path(self, name: str = "specification") -> Path:
        """Retourne le chemin d'une spec."""
        return self.specs_dir / f"{name}.txt"

    # === Méthodes de sauvegarde features ===

    def save_train_features(self, df, engine: str = "pyarrow") -> Path:
        """
        Sauvegarde le DataFrame train avec features.

        Args:
            df: DataFrame pandas avec les features
            engine: Engine parquet ("pyarrow" ou "fastparquet")

        Returns:
            Chemin du fichier sauvegardé
        """
        df.to_parquet(self.train_features_path, engine=engine, index=False)
        self.log(f"Train features sauvegardées: {len(df)} lignes, {len(df.columns)} colonnes")
        return self.train_features_path

    def save_test_features(self, df, engine: str = "pyarrow") -> Path:
        """
        Sauvegarde le DataFrame test avec features.

        Args:
            df: DataFrame pandas avec les features
            engine: Engine parquet ("pyarrow" ou "fastparquet")

        Returns:
            Chemin du fichier sauvegardé
        """
        df.to_parquet(self.test_features_path, engine=engine, index=False)
        self.log(f"Test features sauvegardées: {len(df)} lignes, {len(df.columns)} colonnes")
        return self.test_features_path

    def save_transformed_dataset(self, df_train, df_test=None) -> Path:
        """
        Sauvegarde le dataset transformé en CSV dans dataset_fe/.

        Args:
            df_train: DataFrame train avec les features transformées
            df_test: DataFrame test optionnel

        Returns:
            Chemin du fichier train sauvegardé
        """
        df_train.to_csv(self.train_fe_csv_path, index=False)
        self.log(f"Train FE CSV sauvegardé: {len(df_train)} lignes, {len(df_train.columns)} colonnes")

        if df_test is not None:
            df_test.to_csv(self.test_fe_csv_path, index=False)
            self.log(f"Test FE CSV sauvegardé: {len(df_test)} lignes, {len(df_test.columns)} colonnes")

        return self.train_fe_csv_path

    def save_feature_columns(self, columns: List[str], original_columns: List[str]) -> Path:
        """
        Sauvegarde la liste des colonnes de features.

        Args:
            columns: Liste des colonnes après feature engineering
            original_columns: Liste des colonnes originales

        Returns:
            Chemin du fichier sauvegardé
        """
        data = {
            "original_columns": original_columns,
            "feature_columns": columns,
            "new_features": [c for c in columns if c not in original_columns],
            "n_original": len(original_columns),
            "n_features": len(columns),
            "n_new": len(columns) - len(original_columns),
        }
        return self.save_json(data, self.feature_columns_path)

    def save_pipeline(self, pipeline) -> Path:
        """
        Sauvegarde le pipeline de transformation.

        Args:
            pipeline: Pipeline sklearn ou objet sérialisable

        Returns:
            Chemin du fichier sauvegardé
        """
        import joblib
        joblib.dump(pipeline, self.feature_pipeline_path)
        self.log(f"Pipeline sauvegardé: {self.feature_pipeline_path}")
        return self.feature_pipeline_path

    def load_pipeline(self):
        """
        Charge le pipeline de transformation.

        Returns:
            Pipeline chargé
        """
        import joblib
        return joblib.load(self.feature_pipeline_path)

    # === Méthodes de sauvegarde LLMFE ===

    def save_llmfe_sample(self, sample_order: int, function_str: str, score: Optional[float]) -> Path:
        """
        Sauvegarde un sample LLMFE.

        Args:
            sample_order: Numéro de l'itération
            function_str: Code de la fonction générée
            score: Score d'évaluation (ou None si échec)

        Returns:
            Chemin du fichier créé
        """
        from datetime import datetime
        content = {
            "sample_order": sample_order,
            "function": function_str,
            "score": score,
            "timestamp": datetime.now().isoformat()
        }
        return self.save_json(content, self.get_llmfe_sample_path(sample_order))

    def save_llmfe_best_model(self, model_info: Dict[str, Any]) -> Path:
        """Sauvegarde les infos du meilleur modèle LLMFE."""
        from datetime import datetime
        model_info["saved_at"] = datetime.now().isoformat()
        return self.save_json(model_info, self.llmfe_best_model_path)

    def save_llmfe_scores(self, scores: List[Dict[str, Any]]) -> Path:
        """Sauvegarde tous les scores LLMFE."""
        return self.save_json(scores, self.llmfe_all_scores_path)

    # === Méthodes de sauvegarde specs ===

    def save_spec(self, content: str, name: str = "specification") -> Path:
        """
        Sauvegarde une spécification.

        Args:
            content: Contenu de la spec
            name: Nom du fichier (sans extension)

        Returns:
            Chemin du fichier créé
        """
        path = self.get_spec_path(name)
        path.write_text(content, encoding="utf-8")
        self.log(f"Spec sauvegardée: {path}")
        return path

    def read_spec(self, name: str = "specification") -> str:
        """Lit une spec existante."""
        path = self.get_spec_path(name)
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Spec non trouvée: {path}")

    def read_prompt(self, prompt_type: str, part: str) -> str:
        """
        Lit un fichier de prompt pour LLMFE.

        Args:
            prompt_type: Type de prompt ("operations" ou "domain")
            part: Partie du prompt ("head" ou "tail")

        Returns:
            Contenu du fichier de prompt

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        # Chercher dans le dossier llmfe/prompts relatif au module
        llmfe_prompts_dir = Path(__file__).parent / "llmfe" / "prompts"
        prompt_path = llmfe_prompts_dir / f"{prompt_type}_{part}.txt"

        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")

        raise FileNotFoundError(f"Prompt non trouvé: {prompt_path}")

    # === Métadonnées ===

    def save_fe_metadata(
        self,
        dataset_name: str,
        target_col: str,
        n_rows_train: int,
        n_rows_test: int,
        n_original_features: int,
        n_final_features: int,
        transforms_applied: List[str],
        llmfe_used: bool = False,
        llmfe_best_score: Optional[float] = None,
    ) -> Path:
        """
        Sauvegarde les métadonnées spécifiques au Feature Engineering.

        Args:
            dataset_name: Nom du dataset
            target_col: Colonne cible
            n_rows_train: Nombre de lignes train
            n_rows_test: Nombre de lignes test
            n_original_features: Nombre de features originales
            n_final_features: Nombre de features après FE
            transforms_applied: Liste des transformations appliquées
            llmfe_used: Si LLMFE a été utilisé
            llmfe_best_score: Meilleur score LLMFE (si applicable)

        Returns:
            Chemin du fichier de métadonnées
        """
        metadata = {
            "dataset_name": dataset_name,
            "target_column": target_col,
            "n_rows_train": n_rows_train,
            "n_rows_test": n_rows_test,
            "n_original_features": n_original_features,
            "n_final_features": n_final_features,
            "n_new_features": n_final_features - n_original_features,
            "transforms_applied": transforms_applied,
            "llmfe": {
                "used": llmfe_used,
                "best_score": llmfe_best_score,
            } if llmfe_used else None,
        }
        return self.save_metadata(metadata)
