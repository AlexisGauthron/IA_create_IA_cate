# path_config.py
"""
Gestion centralisée des chemins et fichiers pour LLMFE.
Tous les fichiers générés sont organisés dans un dossier unique par projet.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class LLMFEPathConfig:
    """
    Gestion centralisée des chemins pour LLMFE.

    Structure générée:
    {output_root}/{project_name}/llmfe/
    ├── logs/
    │   └── samples/           ← JSON des samples générés
    ├── specs/                 ← Spec générée pour ce projet
    ├── results/               ← Résultats finaux
    └── tensorboard/           ← Logs TensorBoard
    """

    # Nom du projet (définit le dossier de sortie)
    project_name: str

    # Racine de sortie (par défaut: Creation/)
    output_root: Path = field(default_factory=lambda: Path("Creation"))

    # Racine du module LLMFE (pour les ressources statiques comme les prompts)
    module_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Run ID optionnel pour différencier plusieurs exécutions
    run_id: Optional[str] = None

    # Chemins calculés (initialisés dans __post_init__)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Initialise tous les chemins."""
        self.output_root = Path(self.output_root)
        self.module_root = Path(self.module_root)

        # Générer un run_id si non fourni
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Dossier principal du projet
        self.project_dir = self.output_root / self.project_name / "llmfe"

        # Dossier du run actuel
        self.run_dir = self.project_dir / "runs" / self.run_id

        # Sous-dossiers de sortie (dans le run)
        self.logs_dir = self.run_dir / "logs"
        self.samples_dir = self.run_dir / "samples"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.results_dir = self.run_dir / "results"

        # Dossiers partagés (au niveau projet)
        self.specs_dir = self.project_dir / "specs"

        # Dossiers de ressources statiques (dans le module)
        self.prompts_dir = self.module_root / "prompts"

        self._initialized = True

    def create_directories(self) -> "LLMFEPathConfig":
        """Crée tous les dossiers nécessaires."""
        directories = [
            self.project_dir,
            self.run_dir,
            self.logs_dir,
            self.samples_dir,
            self.tensorboard_dir,
            self.results_dir,
            self.specs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Sauvegarder la config du run
        self._save_run_config()

        return self  # Pour le chaînage

    def _save_run_config(self):
        """Sauvegarde la configuration du run."""
        config_path = self.run_dir / "run_config.json"
        config_data = {
            "project_name": self.project_name,
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "paths": self.to_dict()
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    # ============ PROMPTS (lecture seule) ============

    def get_prompt_path(self, prompt_type: str, part: str) -> Path:
        """
        Retourne le chemin d'un fichier prompt.

        Args:
            prompt_type: 'domain' ou 'operations'
            part: 'head' ou 'tail'

        Returns:
            Path vers le fichier prompt
        """
        return self.prompts_dir / f"{prompt_type}_{part}.txt"

    def read_prompt(self, prompt_type: str, part: str) -> str:
        """
        Lit et retourne le contenu d'un prompt.

        Args:
            prompt_type: 'domain' ou 'operations'
            part: 'head' ou 'tail'

        Returns:
            Contenu du fichier prompt

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        path = self.get_prompt_path(prompt_type, part)
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Prompt non trouvé: {path}")

    def prompt_exists(self, prompt_type: str, part: str) -> bool:
        """Vérifie si un fichier prompt existe."""
        return self.get_prompt_path(prompt_type, part).exists()

    # ============ SPECS (lecture/écriture) ============

    def get_spec_path(self, name: str = "specification") -> Path:
        """Retourne le chemin de la spec du projet."""
        return self.specs_dir / f"{name}.txt"

    def write_spec(self, content: str, name: str = "specification") -> Path:
        """
        Écrit une spec et retourne son chemin.

        Args:
            content: Contenu de la spec
            name: Nom du fichier (sans extension)

        Returns:
            Path du fichier créé
        """
        path = self.get_spec_path(name)
        path.write_text(content, encoding="utf-8")
        return path

    def read_spec(self, name: str = "specification") -> str:
        """Lit une spec existante."""
        path = self.get_spec_path(name)
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Spec non trouvée: {path}")

    def spec_exists(self, name: str = "specification") -> bool:
        """Vérifie si une spec existe."""
        return self.get_spec_path(name).exists()

    # ============ SAMPLES (écriture) ============

    def get_sample_path(self, sample_order: int) -> Path:
        """Retourne le chemin d'un fichier sample."""
        return self.samples_dir / f"sample_{sample_order:04d}.json"

    def write_sample(self, sample_order: int, function_str: str, score: Optional[float]) -> Path:
        """
        Écrit un sample JSON.

        Args:
            sample_order: Numéro de l'itération
            function_str: Code de la fonction générée
            score: Score d'évaluation (ou None si échec)

        Returns:
            Path du fichier créé
        """
        path = self.get_sample_path(sample_order)
        content = {
            "sample_order": sample_order,
            "function": function_str,
            "score": score,
            "timestamp": datetime.now().isoformat()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        return path

    def read_sample(self, sample_order: int) -> Dict[str, Any]:
        """Lit un sample existant."""
        path = self.get_sample_path(sample_order)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_samples(self) -> list:
        """Liste tous les samples générés."""
        return sorted(self.samples_dir.glob("sample_*.json"))

    # ============ RESULTS (écriture) ============

    def get_results_path(self, filename: str) -> Path:
        """Retourne le chemin d'un fichier résultat."""
        return self.results_dir / filename

    def save_best_model(self, model_info: Dict[str, Any]) -> Path:
        """
        Sauvegarde les infos du meilleur modèle.

        Args:
            model_info: Dictionnaire avec les infos du modèle

        Returns:
            Path du fichier créé
        """
        path = self.results_dir / "best_model.json"
        model_info["saved_at"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        return path

    def save_all_scores(self, scores: list) -> Path:
        """Sauvegarde tous les scores dans un fichier JSON."""
        path = self.results_dir / "all_scores.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        return path

    def save_summary(self, summary: Dict[str, Any]) -> Path:
        """Sauvegarde un résumé de l'exécution."""
        path = self.results_dir / "summary.json"
        summary["saved_at"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return path

    # ============ TENSORBOARD ============

    def get_tensorboard_dir(self) -> Path:
        """Retourne le dossier TensorBoard."""
        return self.tensorboard_dir

    # ============ UTILITAIRES ============

    def __str__(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  LLMFE Path Configuration                                    ║
╠══════════════════════════════════════════════════════════════╣
║  Project : {self.project_name:<48} ║
║  Run ID  : {self.run_id:<48} ║
╠══════════════════════════════════════════════════════════════╣
║  Output Structure:                                           ║
║  {str(self.project_dir):<58} ║
║  ├── specs/           (specifications)                       ║
║  └── runs/{self.run_id}/
║      ├── samples/     (JSON samples)                         ║
║      ├── tensorboard/ (TensorBoard logs)                     ║
║      ├── results/     (final results)                        ║
║      └── logs/        (execution logs)                       ║
╠══════════════════════════════════════════════════════════════╣
║  Resources (module):                                         ║
║  └── prompts/         {str(self.prompts_dir):<38} ║
╚══════════════════════════════════════════════════════════════╝
"""

    def to_dict(self) -> Dict[str, str]:
        """Exporte la config en dictionnaire de strings."""
        return {
            "project_name": self.project_name,
            "run_id": self.run_id,
            "project_dir": str(self.project_dir),
            "run_dir": str(self.run_dir),
            "logs_dir": str(self.logs_dir),
            "samples_dir": str(self.samples_dir),
            "tensorboard_dir": str(self.tensorboard_dir),
            "results_dir": str(self.results_dir),
            "specs_dir": str(self.specs_dir),
            "prompts_dir": str(self.prompts_dir),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "LLMFEPathConfig":
        """Crée une instance depuis un dictionnaire."""
        return cls(
            project_name=data["project_name"],
            output_root=Path(data.get("output_root", "Creation")),
            run_id=data.get("run_id"),
        )

    def get_latest_run(self) -> Optional[Path]:
        """Retourne le dossier du dernier run."""
        runs_dir = self.project_dir / "runs"
        if not runs_dir.exists():
            return None
        runs = sorted(runs_dir.iterdir(), reverse=True)
        return runs[0] if runs else None
