# src/llm/business_clarification_bot.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
import json

from src.helper.ollama_llm import OllamaClient
import src.analyse.metier.prompt_metier as prompt_metier
from src.analyse.helper.helper_json_safe import make_json_safe


def normalize_string_whitespace(obj: Any) -> Any:
    """
    Parcourt récursivement une structure JSON-like (dict / list / scalaires)
    et remplace les retours à la ligne dans les chaînes par des espaces simples.
    Utile pour éviter d'avoir des '\\n' dans le JSON envoyé au LLM.
    """
    if isinstance(obj, str):
        s = obj.replace("\r\n", " ").replace("\n", " ")
        # Normalise les espaces multiples
        s = " ".join(s.split())
        return s

    if isinstance(obj, dict):
        return {k: normalize_string_whitespace(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [normalize_string_whitespace(v) for v in obj]

    return obj


@dataclass
class BusinessClarificationBot:
    """
    Chatbot LLM qui, à partir de l'analyse statistique du dataset,
    pose des questions métier pour clarifier :
    - l'objectif business,
    - les enjeux d'erreur,
    - la définition des catégories / labels,
    - les contraintes (temps réel, fairness, etc.).

    Paramètres
    ----------
    stats :
        - soit un dict Python représentant le snapshot (ex : FEDatasetSnapshotForLLM.to_llm_payload()),
        - soit une string JSON déjà prête.
    llm :
        Client Ollama (ou autre) qui expose une méthode .chat(messages: List[dict]) -> str
    """

    stats: Union[Dict[str, Any], str]
    llm: OllamaClient
    messages: List[Dict[str, str]] = field(default_factory=list)

    # sera rempli par _init_system_message
    stats_json: str | None = None

    def __post_init__(self) -> None:
        self._init_system_message()

    # ---------------------------------------------------------
    # Initialisation du prompt système avec le snapshot JSON
    # ---------------------------------------------------------
    def _init_system_message(self) -> None:
        """
        Prépare :
        - le message système (instructions métier),
        - la version JSON minifiée du snapshot (self.stats_json),
        sans encore l'envoyer : ce sera fait dans ask_next().
        """

        # 1) Normalisation -> obtenir un dict JSON-safe si possible
        if isinstance(self.stats, str):
            # On essaie de parser la string comme du JSON
            try:
                raw = json.loads(self.stats)
            except json.JSONDecodeError:
                # Ce n'est pas du JSON strict : on ne peut pas le compacter,
                # on le passera tel quel au LLM comme "texte brut".
                safe_stats = None
                snapshot_text = self.stats
            else:
                safe_stats = make_json_safe(raw)
                snapshot_text = None
        else:
            # Dict Python (snapshot) -> JSON-safe
            safe_stats = make_json_safe(self.stats)
            snapshot_text = None

        # 2) Si on a un dict JSON-safe, on applique la pipeline de compactage
        if safe_stats is not None:

            # 2.2 Normalisation des strings (optionnelle, pour virer \n dans les textes)
            compact_payload = normalize_string_whitespace(safe_stats)

            # 2.3 Minification JSON (aucun espace superflu)
            stats_json_min = json.dumps(
                compact_payload,
                ensure_ascii=False,
                separators=(",", ":"),  # pas d'espace après , ni :
            )

            # 2.4 Garde-fou sur la taille en caractères (optionnel)
            # max_chars = 8000
            # if len(stats_json_min) > max_chars:
            #     stats_json_min = stats_json_min[:max_chars] + "...(tronqué)"

            # ce qu'on utilisera réellement pour le LLM
            self.stats_json = stats_json_min

        else:
            # Cas où self.stats n'était pas du JSON parsable :
            # on garde la string brute (snapshot_text) telle quelle.
            self.stats_json = snapshot_text

        # 3) Construction du message système avec ton prompt métier
        #    (prompt_metier.build_system_content() doit renvoyer le SYSTEM_PROMPT à utiliser)
        system_content = prompt_metier.build_system_content()

        self.messages.append({"role": "system", "content": system_content})

        print(f"[DEBUG] System message initialisé.")
        print(f"\n[INFO] Taille snapshot LLM compacté (JSON minifié) : {len(self.stats_json)} caractères.\n")
        print(f"\n[INFO] Taille contexte LLM : {len(self.messages)} caractères.\n")
        print(f"[DEBUG] stats_json (début) : {str(self.stats_json)[:200]}...\n")

    # ---------------------------------------------------------
    # Boucle de conversation
    # ---------------------------------------------------------
    def ask_next(self, user_answer: str | None = None) -> str:
        """
        - Premier appel (user_answer=None) :
            → envoie le snapshot JSON compressé au LLM.
        - Appels suivants (user_answer str) :
            → envoie la réponse de l'utilisateur et demande la prochaine question.
        """
        if user_answer is None:
            # Premier tour : on fournit au LLM le snapshot compressé
            user_content = (
                "Voici le snapshot JSON compressé du dataset :\n"
                + (self.stats_json or "")
            )
        else:
            # Tours suivants : on fournit la réponse de l'utilisateur
            user_content = (
                f"Voici ma réponse à ta dernière question :\n{user_answer}\n\n"
                "En te basant sur toute la conversation précédente et sur l'analyse statistique, "
                "pose maintenant LA prochaine question métier la plus utile. "
                "Rappelle-toi : une seule question, format 'Q: ...'."
            )

        self.messages.append({"role": "user", "content": user_content})
        print(f"[DEBUG] Messages envoyés au LLM : {self.messages}\n")

        answer = self.llm.chat(self.messages)
        self.messages.append({"role": "assistant", "content": answer})

        return answer
