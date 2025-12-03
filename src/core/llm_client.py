# src/core/llm_client.py
"""
Client LLM unifié pour Ollama et OpenAI.
"""
from __future__ import annotations
from typing import List, Dict, Literal, Optional
import os
import time
import logging
import requests

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)

Provider = Literal["ollama", "openai"]

# =============================================================================
# Exceptions personnalisées
# =============================================================================

class LLMError(Exception):
    """Classe de base pour les erreurs LLM."""
    pass


class LLMTimeoutError(LLMError):
    """Le LLM n'a pas répondu dans le temps imparti."""
    pass


class LLMConnectionError(LLMError):
    """Erreur de connexion au LLM (réseau, serveur down, etc.)."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit atteint (trop de requêtes)."""
    pass


# =============================================================================
# Configuration retry
# =============================================================================

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # secondes (backoff exponentiel : 2s, 4s, 8s)


class OllamaClient:
    """
    Client LLM unifié pour :
      - Ollama (local) via /api/chat
      - OpenAI via l'API officielle (chat.completions)

    Usage typique :

        # === Ollama (comme avant) ===
        client = OllamaClient(
            provider="ollama",
            model="mistral",
        )

        # === OpenAI ===
        client = OllamaClient(
            provider="openai",
            model="gpt-4o-mini",
        )

        response = client.chat([
            {"role": "system", "content": "Tu es un expert en feature engineering."},
            {"role": "user", "content": "Propose des features pour ce dataset."},
        ])

    Les clés API sont chargées depuis:
      1. Le paramètre `openai_api_key` si fourni
      2. Le fichier .env (via src.core.config)
      3. La variable d'environnement OPENAI_API_KEY
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        provider: Optional[Provider] = None,
        # --- Ollama ---
        base_url: Optional[str] = None,
        # --- OpenAI ---
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        # --- Params communs ---
        temperature: float = 0.2,
        max_tokens: int = 8024,
        format_llm: Optional[str] = None,
    ) -> None:
        # Charger les settings pour les valeurs par défaut
        from src.core.config import settings

        # Appliquer les valeurs par défaut depuis la config
        self.provider: Provider = provider or settings.default_provider
        self.model = model or settings.default_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.format_llm = format_llm

        # --- OpenAI client ---
        if self.provider == "openai":
            # Priorité : paramètre > config (.env) > variable d'environnement
            api_key = openai_api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY manquant. "
                    "Configurez-la dans le fichier .env ou passez `openai_api_key=`."
                )

            client_kwargs = {"api_key": api_key}
            if openai_base_url:
                client_kwargs["base_url"] = openai_base_url

            self._openai_client = OpenAI(**client_kwargs)
        else:
            self._openai_client = None

    # ------------------------------------------------------------------
    # Méthode principale : même signature que ton ancien client
    # ------------------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages = [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          ...
        ]
        """
        if self.provider == "ollama":
            return self._chat_ollama(messages)
        elif self.provider == "openai":
            return self._chat_openai(messages)
        else:
            raise ValueError(f"Provider inconnu : {self.provider!r}")

    # ------------------------------------------------------------------
    # Implémentation Ollama (avec retry et gestion d'erreurs)
    # ------------------------------------------------------------------
    def _chat_ollama(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "format": self.format_llm,
        }

        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(self.base_url, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                content = data["message"]["content"]

                # Vérifier que la réponse n'est pas vide
                if not content or content.strip() == "":
                    logger.warning(f"Réponse vide du LLM (tentative {attempt + 1}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BASE_DELAY * (attempt + 1))
                        continue
                    raise LLMError("Le LLM a retourné une réponse vide après plusieurs tentatives")

                return content

            except requests.Timeout as e:
                last_error = e
                logger.warning(f"Timeout Ollama (tentative {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (attempt + 1))
                else:
                    raise LLMTimeoutError(
                        f"Le LLM Ollama n'a pas répondu après {MAX_RETRIES} tentatives (timeout 120s)"
                    ) from e

            except requests.ConnectionError as e:
                last_error = e
                logger.warning(f"Erreur connexion Ollama (tentative {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (attempt + 1))
                else:
                    raise LLMConnectionError(
                        f"Impossible de se connecter à Ollama ({self.base_url}). "
                        f"Vérifiez que le serveur est démarré."
                    ) from e

            except requests.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response else None
                logger.warning(f"Erreur HTTP Ollama {status_code} (tentative {attempt + 1}/{MAX_RETRIES})")

                # Erreurs 5xx = serveur, on peut retry
                if status_code and 500 <= status_code < 600:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BASE_DELAY * (attempt + 1))
                        continue

                # Erreurs 4xx = client, pas de retry
                raise LLMConnectionError(
                    f"Erreur HTTP {status_code} de Ollama: {e}"
                ) from e

        # Ne devrait pas arriver, mais au cas où
        raise LLMError(f"Erreur inattendue après {MAX_RETRIES} tentatives") from last_error

    # ------------------------------------------------------------------
    # Implémentation OpenAI (avec retry et gestion d'erreurs)
    # ------------------------------------------------------------------
    def _chat_openai(self, messages: List[Dict[str, str]]) -> str:
        if self._openai_client is None:
            raise RuntimeError("Client OpenAI non initialisé.")

        # Gestion optionnelle du format JSON (si demandé)
        response_format = None
        if self.format_llm == "json":
            response_format = {"type": "json_object"}

        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                completion = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    response_format=response_format,
                )

                msg = completion.choices[0].message
                content = msg.content or ""

                # Vérifier que la réponse n'est pas vide
                if not content or content.strip() == "":
                    logger.warning(f"Réponse vide OpenAI (tentative {attempt + 1}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BASE_DELAY * (attempt + 1))
                        continue
                    raise LLMError("OpenAI a retourné une réponse vide après plusieurs tentatives")

                return content

            except APITimeoutError as e:
                last_error = e
                logger.warning(f"Timeout OpenAI (tentative {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (attempt + 1))
                else:
                    raise LLMTimeoutError(
                        f"OpenAI n'a pas répondu après {MAX_RETRIES} tentatives"
                    ) from e

            except RateLimitError as e:
                last_error = e
                logger.warning(f"Rate limit OpenAI (tentative {attempt + 1}/{MAX_RETRIES}): {e}")
                # Rate limit = attendre plus longtemps
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (attempt + 1) * 2  # Double le délai
                    logger.info(f"Attente de {delay}s avant retry...")
                    time.sleep(delay)
                else:
                    raise LLMRateLimitError(
                        f"Rate limit OpenAI atteint après {MAX_RETRIES} tentatives. "
                        f"Attendez quelques minutes avant de réessayer."
                    ) from e

            except APIError as e:
                last_error = e
                logger.warning(f"Erreur API OpenAI (tentative {attempt + 1}/{MAX_RETRIES}): {e}")
                # Erreurs serveur (5xx) = retry, autres = stop
                if hasattr(e, 'status_code') and e.status_code and 500 <= e.status_code < 600:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BASE_DELAY * (attempt + 1))
                        continue
                raise LLMConnectionError(f"Erreur API OpenAI: {e}") from e

        # Ne devrait pas arriver
        raise LLMError(f"Erreur inattendue après {MAX_RETRIES} tentatives") from last_error
