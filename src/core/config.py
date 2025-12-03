# src/core/config.py
"""
Gestion centralisée de la configuration et des clés API.

Usage:
    from src.core.config import settings

    # Accéder aux clés
    api_key = settings.openai_api_key

    # Ou utiliser get_api_key() avec fallback
    api_key = settings.get_api_key("openai")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Charger les variables d'environnement depuis .env
try:
    from dotenv import load_dotenv

    # Chercher le fichier .env à la racine du projet
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        print(f"[CONFIG] Fichier .env chargé depuis {env_path}")
    else:
        # Essayer aussi le répertoire courant
        load_dotenv()
except ImportError:
    print("[CONFIG] python-dotenv non installé. Utilisation des variables d'environnement système uniquement.")


@dataclass
class Settings:
    """Configuration centralisée de l'application."""

    # === Clés API ===
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    huggingface_api_key: Optional[str] = field(default_factory=lambda: os.getenv("HUGGINGFACE_API_KEY"))

    # === URLs ===
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/chat"))

    # === Defaults LLM ===
    default_provider: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", "openai"))
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"))

    # === Chemins ===
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "outputs")))
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "Modeles")))

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Récupère la clé API pour un provider donné.

        Args:
            provider: "openai", "huggingface", etc.

        Returns:
            La clé API ou None si non trouvée.
        """
        key_mapping = {
            "openai": self.openai_api_key,
            "huggingface": self.huggingface_api_key,
            "hf": self.huggingface_api_key,
        }
        return key_mapping.get(provider.lower())

    def require_api_key(self, provider: str) -> str:
        """
        Récupère la clé API, lève une erreur si non trouvée.

        Args:
            provider: "openai", "huggingface", etc.

        Returns:
            La clé API.

        Raises:
            ValueError: Si la clé n'est pas configurée.
        """
        key = self.get_api_key(provider)
        if not key:
            raise ValueError(
                f"Clé API pour '{provider}' non trouvée. "
                f"Configurez {provider.upper()}_API_KEY dans votre fichier .env "
                f"ou comme variable d'environnement."
            )
        return key

    def is_configured(self, provider: str) -> bool:
        """Vérifie si un provider est configuré."""
        return self.get_api_key(provider) is not None

    def to_dict(self) -> Dict[str, Any]:
        """Retourne la configuration sous forme de dictionnaire (sans les clés sensibles)."""
        return {
            "openai_configured": self.openai_api_key is not None,
            "huggingface_configured": self.huggingface_api_key is not None,
            "ollama_base_url": self.ollama_base_url,
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "models_dir": str(self.models_dir),
        }

    def __repr__(self) -> str:
        return f"Settings({self.to_dict()})"


# Instance globale (singleton)
settings = Settings()


# === Fonctions utilitaires ===

def get_openai_key() -> str:
    """Raccourci pour obtenir la clé OpenAI."""
    return settings.require_api_key("openai")


def get_huggingface_key() -> str:
    """Raccourci pour obtenir la clé HuggingFace."""
    return settings.require_api_key("huggingface")


def is_openai_configured() -> bool:
    """Vérifie si OpenAI est configuré."""
    return settings.is_configured("openai")


def is_ollama_available() -> bool:
    """Vérifie si Ollama est accessible."""
    import requests
    try:
        response = requests.get(
            settings.ollama_base_url.replace("/api/chat", "/api/tags"),
            timeout=2
        )
        return response.status_code == 200
    except Exception:
        return False
