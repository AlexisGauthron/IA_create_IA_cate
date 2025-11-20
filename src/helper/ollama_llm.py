# src/llm/ollama_client.py
from __future__ import annotations
from typing import List, Dict, Literal, Optional
import os
import requests

from openai import OpenAI


Provider = Literal["ollama", "openai"]


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
            model="gpt-4.1-mini",
        )

        response = client.chat([
            {"role": "system", "content": "Tu es un expert en feature engineering."},
            {"role": "user", "content": "Propose des features pour ce dataset."},
        ])
    """

    def __init__(
        self,
        model: str = "mistral",
        *,
        provider: Provider = "ollama",
        # --- Ollama ---
        base_url: str = "http://localhost:11434/api/chat",
        # --- OpenAI ---
        openai_api_key: Optional[str] = "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",
        openai_base_url: Optional[str] = None,  # utile si tu utilises un proxy / endpoint custom
        # --- Params communs ---
        temperature: float = 0.2,
        max_tokens: int = 8024,
        format_llm: Optional[str] = None,  # ex: "json" si tu veux forcer du JSON
    ) -> None:
        self.model = model
        self.provider: Provider = provider
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.format_llm = format_llm

        # --- OpenAI client ---
        if self.provider == "openai":
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY manquant. "
                    "Passe `openai_api_key=` ou définis la variable d'environnement OPENAI_API_KEY."
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
    # Implémentation Ollama (comme avant)
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
        resp = requests.post(self.base_url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Format de réponse standard de /api/chat :
        # {"message": {"role": "assistant", "content": "..."} , ...}
        return data["message"]["content"]

    # ------------------------------------------------------------------
    # Implémentation OpenAI (chat.completions)
    # ------------------------------------------------------------------
    def _chat_openai(self, messages: List[Dict[str, str]]) -> str:
        if self._openai_client is None:
            raise RuntimeError("Client OpenAI non initialisé.")

        # Gestion optionnelle du format JSON (si demandé)
        response_format = None
        if self.format_llm == "json":
            response_format = {"type": "json_object"}

        completion = self._openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            # nouveau param pour limiter les tokens de sortie sur les modèles récents
            max_completion_tokens=self.max_tokens,
            response_format=response_format,
        )

        msg = completion.choices[0].message
        content = msg.content or ""

        # Si tu veux, tu peux ici parser le JSON si `format_llm == "json"`
        # par exemple :
        # if self.format_llm == "json":
        #     import json
        #     return json.loads(content)

        return content
