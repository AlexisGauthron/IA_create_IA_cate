# src/core/base_path_config.py
"""
Classe de base pour la gestion des chemins de sortie.
Tous les PathConfig des modules héritent de cette classe.

Architecture:
    Settings (config.py)     → Définit output_dir, models_dir, data_dir
         │
         ▼
    BasePathConfig           → Lit depuis Settings, gère timestamp/logs
         │
    ┌────┴────┬──────────────┐
    ▼         ▼              ▼
 Analyse   AutoML         LLMFE
PathConfig PathConfig   PathConfig
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


class BasePathConfig(ABC):
    """
    Classe abstraite de base pour la gestion des chemins.

    Hérite de cette classe pour créer un PathConfig spécifique à un module.
    Les chemins racines sont lus depuis Settings pour centraliser la configuration.

    Structure standard des outputs:
    {output_dir}/
    └── {project_name}/
        └── {module_name}/
            ├── {sous-dossiers spécifiques au module}
            └── logs/

    Note: Le timestamp n'est plus utilisé dans la structure des dossiers.
    Il est stocké uniquement dans les métadonnées pour traçabilité.
    """

    # Nom du module (à définir dans les sous-classes)
    MODULE_NAME: str = "base"

    def __init__(
        self,
        project_name: str,
        base_dir: Optional[str | Path] = None,
    ):
        """
        Initialise la configuration des chemins.

        Args:
            project_name: Nom du projet (ex: "titanic", "verbatims")
            base_dir: Dossier racine optionnel (sinon lu depuis Settings)
        """
        # Import ici pour éviter les imports circulaires
        from src.core.config import settings

        self.project_name = project_name
        # Timestamp pour les métadonnées uniquement (pas dans la structure des dossiers)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._settings = settings

        # Déterminer le dossier racine
        if base_dir is not None:
            self._base_dir = Path(base_dir)
        else:
            self._base_dir = settings.output_dir

        # Construire le chemin du projet
        # Structure: {output_dir}/{project_name}/{module_name}/
        self.project_dir = self._base_dir / project_name / self.MODULE_NAME

        # Dossier de logs (commun à tous les modules)
        self.logs_dir = self.project_dir / "logs"

        # Note: _create_directories() est appelé par les sous-classes après
        # avoir défini leurs attributs spécifiques

    # === Méthodes abstraites (à implémenter dans les sous-classes) ===

    @abstractmethod
    def _get_subdirectories(self) -> List[Path]:
        """
        Retourne la liste des sous-dossiers spécifiques au module.

        À implémenter dans chaque sous-classe.

        Returns:
            Liste des Path des sous-dossiers à créer
        """
        pass

    @abstractmethod
    def get_all_paths(self) -> Dict[str, str]:
        """
        Retourne tous les chemins configurés pour ce module.

        À implémenter dans chaque sous-classe.

        Returns:
            Dictionnaire {nom: chemin}
        """
        pass

    # === Méthodes communes ===

    def _create_directories(self) -> None:
        """Crée tous les dossiers nécessaires."""
        # Créer le dossier de logs (commun)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Créer les sous-dossiers spécifiques au module
        for directory in self._get_subdirectories():
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        """Chemin du fichier de log principal."""
        return self.logs_dir / f"{self.MODULE_NAME}.log"

    @property
    def metadata_path(self) -> Path:
        """Chemin du fichier de métadonnées."""
        return self.project_dir / "metadata.json"

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Ajoute un message au fichier de log.

        Args:
            message: Message à logger
            level: Niveau de log (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}\n"

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

    def save_json(self, data: Dict[str, Any], path: Path, verbose: bool = True) -> Path:
        """
        Sauvegarde un dictionnaire en JSON.

        Args:
            data: Données à sauvegarder
            path: Chemin du fichier
            verbose: Afficher un message de confirmation

        Returns:
            Chemin du fichier sauvegardé
        """
        # Tenter de rendre les données JSON-safe
        try:
            from src.analyse.helper.helper_json_safe import make_json_safe
            data = make_json_safe(data)
        except ImportError:
            pass

        # S'assurer que le dossier parent existe
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        if verbose:
            print(f"[{self.__class__.__name__}] Sauvegardé: {path}")

        self.log(f"Fichier sauvegardé: {path}")
        return path

    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """
        Sauvegarde les métadonnées avec informations communes.

        Args:
            metadata: Dictionnaire de métadonnées spécifiques au module

        Returns:
            Chemin du fichier de métadonnées
        """
        # Ajouter les métadonnées communes
        full_metadata = {
            "project_name": self.project_name,
            "module": self.MODULE_NAME,
            "timestamp": self.timestamp,
            "created_at": datetime.now().isoformat(),
            "output_dir": str(self._base_dir),
            **metadata,  # Métadonnées spécifiques au module
        }
        return self.save_json(full_metadata, self.metadata_path)

    def get_base_paths(self) -> Dict[str, str]:
        """Retourne les chemins de base communs à tous les modules."""
        return {
            "project_dir": str(self.project_dir),
            "logs_dir": str(self.logs_dir),
            "log_file": str(self.log_path),
            "metadata": str(self.metadata_path),
        }

    @classmethod
    def from_existing(cls, project_dir: str | Path) -> "BasePathConfig":
        """
        Crée une instance à partir d'un dossier existant.

        Args:
            project_dir: Chemin vers un dossier de projet existant

        Returns:
            Instance du PathConfig
        """
        project_dir = Path(project_dir)

        # Structure: {base_dir}/{project_name}/{module_name}/
        # module_name = project_dir.name (implicite via MODULE_NAME de la classe)
        project_name = project_dir.parent.name
        base_dir = project_dir.parent.parent

        return cls(
            project_name=project_name,
            base_dir=base_dir,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(project='{self.project_name}', module='{self.MODULE_NAME}', dir='{self.project_dir}')"

    def __str__(self) -> str:
        return str(self.project_dir)
