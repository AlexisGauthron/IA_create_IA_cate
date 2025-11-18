# src/llm/ollama_client.py
from __future__ import annotations
from typing import List, Dict
import requests


class OllamaClient:
    """
    Client minimal pour appeler un modèle (ex: mistral) via Ollama en /api/chat.
    Démarrer Ollama en local puis : `ollama pull mistral` si ce n'est pas déjà fait.
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.2,
        max_tokens: int = 8024,
        format_llm: str = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.format_llm = format_llm

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages = [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          ...
        ]
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },

            "format": self.format_llm
        }
        resp = requests.post(f"{self.base_url}", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Format de réponse standard de /api/chat :
        # {"message": {"role": "assistant", "content": "..."} , ...}
        return data["message"]["content"]
