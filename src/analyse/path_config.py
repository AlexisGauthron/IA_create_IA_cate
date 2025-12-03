# src/analyse/path_config.py
"""
Gestionnaire de chemins pour le module d'analyse.
Hérite de BasePathConfig pour la gestion centralisée.

Structure des dossiers créés:
{output_dir}/
└── {project_name}/
    └── analyse/
        ├── stats/
        │   └── report_stats.json
        ├── full/
        │   └── report_full.json
        ├── agent_llm/
        │   └── conversation.json
        └── logs/
            └── analyse.log

Note: Le timestamp n'est plus dans la structure des dossiers,
il est stocké uniquement dans les métadonnées.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List

from src.core.base_path_config import BasePathConfig


class AnalysePathConfig(BasePathConfig):
    """
    Configuration des chemins pour les sorties d'analyse.

    Hérite de BasePathConfig et ajoute les sous-dossiers spécifiques:
    - stats/   : Rapport statistique (sans LLM)
    - full/    : Rapport complet (avec annotations LLM)
    """

    MODULE_NAME = "analyse"

    def __init__(
        self,
        project_name: str,
        base_dir: Optional[str | Path] = None,
    ):
        """
        Initialise la configuration des chemins pour l'analyse.

        Args:
            project_name: Nom du projet (ex: "titanic", "verbatims")
            base_dir: Dossier racine optionnel (sinon lu depuis Settings.output_dir)
        """
        # Appel du constructeur parent
        super().__init__(
            project_name=project_name,
            base_dir=base_dir,
        )

        # Sous-dossiers spécifiques à l'analyse
        self.stats_dir = self.project_dir / "stats"
        self.full_dir = self.project_dir / "full"
        self.agent_llm_dir = self.project_dir / "agent_llm"

        # Créer les dossiers
        self._create_directories()

    # === Implémentation des méthodes abstraites ===

    def _get_subdirectories(self) -> List[Path]:
        """Retourne les sous-dossiers spécifiques à l'analyse."""
        return [self.stats_dir, self.full_dir, self.agent_llm_dir]

    def get_all_paths(self) -> Dict[str, str]:
        """Retourne tous les chemins configurés."""
        paths = self.get_base_paths()
        paths.update({
            "stats_dir": str(self.stats_dir),
            "full_dir": str(self.full_dir),
            "agent_llm_dir": str(self.agent_llm_dir),
            "stats_report": str(self.stats_report_path),
            "full_report": str(self.full_report_path),
            "conversation": str(self.conversation_path),
        })
        return paths

    # === Chemins des fichiers spécifiques ===

    @property
    def stats_report_path(self) -> Path:
        """Chemin du rapport statistiques (sans annotations LLM)."""
        return self.stats_dir / "report_stats.json"

    @property
    def full_report_path(self) -> Path:
        """Chemin du rapport complet (avec annotations LLM)."""
        return self.full_dir / "report_full.json"

    @property
    def conversation_path(self) -> Path:
        """Chemin du fichier de conversation avec l'agent LLM."""
        return self.agent_llm_dir / "conversation.json"

    # === Méthodes de sauvegarde spécifiques ===

    def save_stats_report(self, report: Dict[str, Any]) -> Path:
        """
        Sauvegarde le rapport statistiques.

        Args:
            report: Dictionnaire du rapport

        Returns:
            Chemin du fichier sauvegardé
        """
        self.log("Sauvegarde du rapport statistiques")
        return self.save_json(report, self.stats_report_path)

    def save_full_report(self, report: Dict[str, Any]) -> Path:
        """
        Sauvegarde le rapport complet avec annotations LLM.

        Args:
            report: Dictionnaire du rapport complet

        Returns:
            Chemin du fichier sauvegardé
        """
        self.log("Sauvegarde du rapport complet avec LLM")
        return self.save_json(report, self.full_report_path)

    def save_analyse_metadata(
        self,
        dataset_name: str,
        target_col: str,
        n_rows: int,
        n_features: int,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        only_stats: bool = False,
    ) -> Path:
        """
        Sauvegarde les métadonnées spécifiques à l'analyse.

        Args:
            dataset_name: Nom du dataset
            target_col: Colonne cible
            n_rows: Nombre de lignes
            n_features: Nombre de features
            provider: Provider LLM utilisé (si applicable)
            model: Modèle LLM utilisé (si applicable)
            only_stats: Si True, analyse stats uniquement

        Returns:
            Chemin du fichier de métadonnées
        """
        metadata = {
            "dataset_name": dataset_name,
            "target_column": target_col,
            "n_rows": n_rows,
            "n_features": n_features,
            "analysis_type": "stats_only" if only_stats else "full_with_llm",
            "llm_config": {
                "provider": provider,
                "model": model,
            } if not only_stats else None,
        }
        return self.save_metadata(metadata)

    def save_conversation(self, conversation_data: Dict[str, Any]) -> Path:
        """
        Sauvegarde la conversation avec l'agent LLM.

        Args:
            conversation_data: Dictionnaire contenant la conversation complète
                Structure attendue:
                {
                    "metadata": {
                        "timestamp": str,
                        "provider": str,
                        "model": str,
                        "project": str
                    },
                    "system_prompt": str,
                    "conversation": [
                        {"role": str, "content": str, "timestamp": str},
                        ...
                    ],
                    "final_report": dict
                }

        Returns:
            Chemin du fichier sauvegardé
        """
        self.log("Sauvegarde de la conversation avec l'agent LLM")
        # S'assurer que le dossier existe
        self.agent_llm_dir.mkdir(parents=True, exist_ok=True)
        return self.save_json(conversation_data, self.conversation_path)

    def has_llm_analysis(self) -> bool:
        """
        Vérifie si l'analyse LLM a déjà été effectuée.

        Returns:
            True si le rapport complet (full) existe, False sinon
        """
        return self.full_report_path.exists()
