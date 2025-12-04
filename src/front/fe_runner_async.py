"""
Wrapper asynchrone pour LLMFE.
Permet de lancer le Feature Engineering dans un thread séparé
pour que Streamlit puisse afficher la progression en temps réel.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner
from src.feature_engineering.path_config import FeatureEngineeringPathConfig


class AsyncFERunner:
    """
    Lance LLMFERunner dans un thread séparé.

    Permet à Streamlit de faire du polling sur les résultats
    pendant que LLMFE s'exécute en arrière-plan.
    """

    def __init__(self, project_name: str):
        """
        Initialise le runner asynchrone.

        Args:
            project_name: Nom du projet (pour les chemins de sortie)
        """
        self.project_name = project_name
        self.runner: LLMFERunner | None = None
        self.path_config: FeatureEngineeringPathConfig | None = None
        self.thread: threading.Thread | None = None
        self.result: dict[str, Any] | None = None
        self.error: Exception | None = None
        self._is_started = False

    def start(
        self,
        df_train,
        target_col: str,
        is_regression: bool = False,
        max_samples: int = 20,
        api_model: str = "gpt-4o-mini",
        use_api: bool = True,
        **kwargs,
    ):
        """
        Lance LLMFE dans un thread séparé.

        Args:
            df_train: DataFrame d'entraînement
            target_col: Colonne cible
            is_regression: True si régression
            max_samples: Nombre max d'itérations
            api_model: Modèle API à utiliser
            use_api: Utiliser l'API OpenAI
            **kwargs: Arguments additionnels pour LLMFERunner.run()
        """
        # Créer le runner et le path_config AVANT de lancer le thread
        # pour pouvoir accéder au dossier samples immédiatement
        self.runner = LLMFERunner(project_name=self.project_name)

        # Initialiser le path_config manuellement pour connaître les chemins
        self.path_config = FeatureEngineeringPathConfig(
            project_name=self.project_name,
        )
        self.runner.path_config = self.path_config

        # Créer les dossiers nécessaires
        self.path_config.llmfe_samples_dir.mkdir(parents=True, exist_ok=True)
        self.path_config.llmfe_results_dir.mkdir(parents=True, exist_ok=True)

        # Préparer les arguments pour le thread
        run_kwargs = {
            "df_train": df_train,
            "target_col": target_col,
            "is_regression": is_regression,
            "max_samples": max_samples,
            "api_model": api_model,
            "use_api": use_api,
            **kwargs,
        }

        # Lancer le thread
        self.thread = threading.Thread(
            target=self._run_in_thread,
            kwargs=run_kwargs,
            daemon=True,  # Le thread sera tué si le programme principal s'arrête
        )
        self.thread.start()
        self._is_started = True

    def _run_in_thread(self, **kwargs):
        """Exécute LLMFE dans le thread."""
        try:
            self.result = self.runner.run(**kwargs)
        except Exception as e:
            self.error = e
            import traceback

            print(f"[AsyncFERunner] Erreur: {e}")
            print(traceback.format_exc())

    def is_running(self) -> bool:
        """Retourne True si LLMFE est en cours d'exécution."""
        if not self._is_started:
            return False
        return self.thread is not None and self.thread.is_alive()

    def is_finished(self) -> bool:
        """Retourne True si LLMFE a terminé (succès ou erreur)."""
        return self._is_started and not self.is_running()

    def has_error(self) -> bool:
        """Retourne True si une erreur s'est produite."""
        return self.error is not None

    def get_samples_dir(self) -> Path | None:
        """Retourne le chemin du dossier samples (disponible dès le start)."""
        if self.path_config is not None:
            return self.path_config.llmfe_samples_dir
        return None

    def get_results_dir(self) -> Path | None:
        """Retourne le chemin du dossier results."""
        if self.path_config is not None:
            return self.path_config.llmfe_results_dir
        return None

    def get_result(self) -> dict[str, Any] | None:
        """Retourne le résultat final (None si pas encore terminé)."""
        return self.result

    def get_error(self) -> Exception | None:
        """Retourne l'erreur si une s'est produite."""
        return self.error

    def wait(self, timeout: float | None = None):
        """Attend la fin du thread."""
        if self.thread is not None:
            self.thread.join(timeout=timeout)
