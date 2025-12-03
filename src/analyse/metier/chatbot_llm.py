# src/llm/business_clarification_bot.py
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Union, Optional
import json

from src.core.llm_client import OllamaClient
import src.analyse.metier.prompt_metier as prompt_metier
from src.analyse.helper.helper_json_safe import make_json_safe

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration limite de contexte
# =============================================================================

# Nombre maximum de messages dans l'historique (hors system)
# Cela correspond à ~10 échanges question/réponse
MAX_CONVERSATION_MESSAGES = 20


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

    # Historique de conversation avec timestamps pour export
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: str | None = None
    start_time: str | None = None

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

        # Stocker le system_prompt pour l'export
        self.system_prompt = system_content
        self.start_time = datetime.now().isoformat()

        self.messages.append({"role": "system", "content": system_content})

        logger.debug("System message initialisé.")
        logger.info(f"Taille snapshot LLM compacté (JSON minifié) : {len(self.stats_json)} caractères.")
        logger.info(f"Taille contexte LLM : {len(self.messages)} messages.")
        logger.debug(f"stats_json (début) : {str(self.stats_json)[:200]}...")




    # ---------------------------------------------------------
    # Gestion de la taille du contexte
    # ---------------------------------------------------------
    def _truncate_history_if_needed(self) -> None:
        """
        Tronque l'historique si trop de messages pour éviter de dépasser
        le contexte du LLM.

        Stratégie : garder [system] + [premier message avec JSON] + N derniers messages
        """
        # messages[0] = system prompt
        # messages[1] = premier user message (contient le JSON stats)
        # messages[2:] = conversation Q&A

        if len(self.messages) <= MAX_CONVERSATION_MESSAGES + 2:
            return  # Pas besoin de tronquer

        # Garder : system (0) + premier message (1) + derniers messages
        n_to_keep = MAX_CONVERSATION_MESSAGES
        self.messages = self.messages[:2] + self.messages[-n_to_keep:]

        logger.info(
            f"Historique tronqué pour rester sous {MAX_CONVERSATION_MESSAGES} messages. "
            f"Nouveau total: {len(self.messages)}"
        )

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

        # Enregistrer le message utilisateur dans l'historique avec timestamp
        self.conversation_history.append({
            "role": "user",
            "content": user_content,
            "timestamp": datetime.now().isoformat()
        })

        # Tronquer l'historique si nécessaire AVANT d'envoyer au LLM
        self._truncate_history_if_needed()

        logger.debug(f"Messages envoyés au LLM : {len(self.messages)} messages")

        answer = self.llm.chat(self.messages)
        self.messages.append({"role": "assistant", "content": answer})

        # Enregistrer la réponse de l'assistant dans l'historique avec timestamp
        self.conversation_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat()
        })

        return answer

    # ---------------------------------------------------------
    # Export de la conversation
    # ---------------------------------------------------------
    def export_conversation(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        project: Optional[str] = None,
        final_report: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Exporte la conversation complète dans un format structuré.

        Args:
            provider: Nom du provider LLM utilisé (ex: "openai", "ollama")
            model: Nom du modèle utilisé (ex: "gpt-4o-mini")
            project: Nom du projet
            final_report: Rapport final généré par le LLM

        Returns:
            Dictionnaire contenant la conversation complète avec métadonnées
        """
        return {
            "metadata": {
                "start_time": self.start_time,
                "end_time": datetime.now().isoformat(),
                "provider": provider,
                "model": model,
                "project": project,
                "total_exchanges": len(self.conversation_history)
            },
            "system_prompt": self.system_prompt,
            "stats_snapshot": self.stats_json,
            "conversation": self.conversation_history,
            "final_report": final_report
        }
