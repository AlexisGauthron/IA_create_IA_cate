"""
Agent de clarification métier pour l'analyse de datasets.

Ce module orchestre la conversation avec le LLM pour enrichir
l'analyse statistique avec des informations métier :
- business_description : description métier du dataset
- final_metric : métrique d'évaluation choisie par le LLM
- final_metric_reason : justification métier de la métrique
- feature_description : description métier de chaque feature

Usage:
    from src.analyse.metier.business_agent import run_business_clarification

    result = run_business_clarification(
        stats_payload=stats_payload,
        provider="openai",
        model="gpt-4o-mini"
    )
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

from src.analyse.helper.compress_data import compact_llm_snapshot_payload
from src.analyse.metier.chatbot_llm import BusinessClarificationBot
from src.analyse.metier.parsing_json import apply_llm_business_annotations
from src.core.llm_client import (
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    OllamaClient,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Commandes utilisateur
# =============================================================================

SKIP_COMMANDS = {"skip", "passe", "suivant", "next"}
DONE_COMMANDS = {"done", "fin", "stop", "terminer", "fini", "terminé", "Arrête"}

SKIP_RESPONSE = "Je ne sais pas, passe à la suite."

DONE_RESPONSE = (
    "L'utilisateur souhaite terminer la session. "
    "Génère maintenant le rapport FINAL avec toutes les informations "
    "collectées jusqu'ici. Utilise des scores de confidence appropriés "
    "pour les champs où tu manques d'informations."
)


# =============================================================================
# Agent principal
# =============================================================================


def run_business_clarification(
    stats_payload: dict[str, Any],
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    interactive: bool = True,
    verbose: bool = True,
    max_questions: int = 10,
) -> dict[str, Any]:
    """
    Lance une session de clarification métier avec le LLM.

    Cette fonction orchestre la conversation entre l'utilisateur et le LLM
    pour enrichir le payload statistique avec des informations métier.

    Args:
        stats_payload: Le JSON stats du dataset (sortie de analyze_dataset_for_fe)
        provider: Provider LLM ("openai" ou "ollama")
        model: Modèle LLM à utiliser (ex: "gpt-4o-mini", "llama3")
        interactive: Si True, demande les réponses via input()
                     Si False, le LLM fait ses hypothèses seul (mode auto)
        verbose: Affiche les messages de progression
        max_questions: Nombre maximum de questions avant de forcer le mode Final

    Returns:
        Le payload enrichi avec les annotations LLM (business_description,
        final_metric, final_metric_reason, feature_descriptions).

    Note:
        Cette fonction ne retourne jamais None. Si l'utilisateur veut arrêter,
        le LLM génère quand même un rapport avec ce qu'il a compris.
    """

    if verbose:
        print("\n" + "=" * 60)
        print("  AGENT DE CLARIFICATION MÉTIER")
        print("=" * 60)
        print(f"  Provider: {provider} | Modèle: {model}")
        print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Compacter le payload pour le LLM (réduction des tokens)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[1/3] Préparation du payload pour le LLM...")

    compact_payload = compact_llm_snapshot_payload(
        payload=stats_payload,
        max_example_values=3,
        max_top_values=3,
        float_ndigits=4,
        feature_engineering=False,  # On ne fait pas de FE ici, juste du métier
    )

    logger.info("Payload compacté pour LLM")

    # -------------------------------------------------------------------------
    # 2. Initialiser le client LLM et le bot de conversation
    # -------------------------------------------------------------------------
    if verbose:
        print("[2/3] Initialisation du client LLM...")

    llm = OllamaClient(model=model, provider=provider, format_llm="json")
    bot = BusinessClarificationBot(stats=compact_payload, llm=llm)

    # -------------------------------------------------------------------------
    # 3. Boucle de conversation
    # -------------------------------------------------------------------------
    if verbose:
        print("[3/3] Démarrage de la conversation...\n")
        _print_help()

    llm_response = None
    llm_error_occurred = False

    # Premier appel : envoie le snapshot et récupère la première question/réponse
    try:
        llm_response = bot.ask_next()

        if verbose:
            print("\n" + "-" * 60)
            print("LLM:", llm_response)
            print("-" * 60)
    except LLMTimeoutError as e:
        logger.error(f"Timeout LLM au premier appel: {e}")
        if verbose:
            print(f"\n[ERREUR] Le LLM n'a pas répondu (timeout): {e}")
            print("[INFO] Retour du payload sans enrichissement LLM")
        llm_error_occurred = True
    except LLMConnectionError as e:
        logger.error(f"Erreur connexion LLM: {e}")
        if verbose:
            print(f"\n[ERREUR] Impossible de se connecter au LLM: {e}")
            print("[INFO] Retour du payload sans enrichissement LLM")
        llm_error_occurred = True
    except LLMRateLimitError as e:
        logger.error(f"Rate limit LLM: {e}")
        if verbose:
            print(f"\n[ERREUR] Rate limit atteint: {e}")
            print("[INFO] Retour du payload sans enrichissement LLM")
        llm_error_occurred = True
    except LLMError as e:
        logger.error(f"Erreur LLM: {e}")
        if verbose:
            print(f"\n[ERREUR] Erreur LLM: {e}")
            print("[INFO] Retour du payload sans enrichissement LLM")
        llm_error_occurred = True

    # Si erreur dès le premier appel, retourner le payload original
    if llm_error_occurred:
        return deepcopy(stats_payload)

    question_count = 0

    while True:
        # Vérifier si on est en mode Final
        if _is_final_mode(llm_response):
            if verbose:
                print("\n[INFO] Mode Final atteint - Rapport généré par le LLM")
            break

        # Vérifier le nombre max de questions
        question_count += 1
        if question_count >= max_questions:
            if verbose:
                print(f"\n[INFO] Limite de {max_questions} questions atteinte")
                print("[INFO] Demande du rapport final au LLM...")

            try:
                llm_response = bot.ask_next(DONE_RESPONSE)
                if verbose:
                    print("\n" + "-" * 60)
                    print("LLM:", llm_response)
                    print("-" * 60)
            except LLMError as e:
                logger.error(f"Erreur LLM lors de la demande finale: {e}")
                if verbose:
                    print(f"\n[ERREUR] Erreur LLM: {e}")
                    print("[INFO] Retour du payload sans enrichissement complet")
                break
            continue

        # Obtenir la réponse de l'utilisateur
        if interactive:
            user_input = _get_user_input()
        else:
            # Mode non-interactif : on laisse le LLM faire ses hypothèses
            user_input = (
                "Je n'ai pas d'information supplémentaire, fais de ton mieux avec tes hypothèses."
            )

        # Traiter les commandes spéciales
        user_response = _process_user_input(user_input, verbose)

        # Envoyer la réponse au LLM
        try:
            llm_response = bot.ask_next(user_response)

            if verbose:
                print("\n" + "-" * 60)
                print("LLM:", llm_response)
                print("-" * 60)
        except LLMError as e:
            logger.error(f"Erreur LLM pendant la conversation: {e}")
            if verbose:
                print(f"\n[ERREUR] Erreur LLM: {e}")
                print("[INFO] Tentative de génération du rapport avec les infos collectées...")

            # Tenter une dernière fois de récupérer un rapport
            try:
                llm_response = bot.ask_next(DONE_RESPONSE)
                if verbose:
                    print("\n" + "-" * 60)
                    print("LLM:", llm_response)
                    print("-" * 60)
            except LLMError:
                if verbose:
                    print(
                        "[ERREUR] Impossible de récupérer un rapport. Retour du payload original."
                    )
                return deepcopy(stats_payload)

    # -------------------------------------------------------------------------
    # 4. Fusionner les annotations LLM avec le payload original
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[INFO] Fusion des annotations LLM avec le payload...")

    enriched_payload = apply_llm_business_annotations(
        snapshot=stats_payload,
        llm_result_raw=llm_response,
        min_confidence=0.6,
    )

    if verbose:
        _print_summary(enriched_payload)

    return enriched_payload


# =============================================================================
# Fonctions utilitaires
# =============================================================================


def _print_help() -> None:
    """Affiche l'aide des commandes disponibles."""
    print("Commandes disponibles:")
    print("  - 'skip'  : Passer cette question")
    print("  - 'done'  : Terminer et générer le rapport final")
    print("")


def _get_user_input() -> str:
    """Récupère l'input utilisateur."""
    try:
        return input("\nTa réponse > ").strip()
    except (EOFError, KeyboardInterrupt):
        # Ctrl+C ou Ctrl+D = on termine proprement
        print("\n[INFO] Interruption détectée - Génération du rapport final...")
        return "done"


def _process_user_input(user_input: str, verbose: bool = True) -> str:
    """
    Traite l'input utilisateur et retourne la réponse à envoyer au LLM.

    Args:
        user_input: La saisie brute de l'utilisateur
        verbose: Afficher les messages

    Returns:
        La réponse formatée à envoyer au LLM
    """
    normalized = user_input.lower().strip()

    # Commande SKIP
    if normalized in SKIP_COMMANDS:
        if verbose:
            print("[INFO] Question passée")
        return SKIP_RESPONSE

    # Commande DONE
    if normalized in DONE_COMMANDS:
        if verbose:
            print("[INFO] Demande de génération du rapport final...")
        return DONE_RESPONSE

    # Réponse normale
    return user_input


def _is_final_mode(llm_response: str) -> bool:
    """
    Vérifie si la réponse du LLM indique le mode Final.

    Le LLM répond en JSON avec "Mode": "Final" quand il a
    suffisamment d'informations pour générer le rapport.

    Stratégie de détection :
    1. D'abord essayer de parser le JSON proprement
    2. Sinon fallback sur regex insensible à la casse
    """
    # 1. Essayer le parsing JSON
    try:
        data = json.loads(llm_response)
        # Gérer "Mode" ou "mode" (insensible à la casse)
        mode = data.get("Mode") or data.get("mode") or ""
        if isinstance(mode, str) and mode.lower() == "final":
            return True
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # 2. Fallback regex insensible à la casse
    # Cherche "Mode": "Final" ou "mode":"final" avec espaces variables
    pattern = r'"[Mm]ode"\s*:\s*"[Ff]inal"'
    if re.search(pattern, llm_response):
        return True

    return False


def _print_summary(payload: dict[str, Any]) -> None:
    """Affiche un résumé du payload enrichi."""
    print("\n" + "=" * 60)
    print("  RÉSUMÉ DE L'ENRICHISSEMENT")
    print("=" * 60)

    context = payload.get("context", {})

    # Business description
    bd = context.get("business_description")
    if bd:
        print("\n📋 Description métier:")
        print(f"   {bd[:100]}..." if len(str(bd)) > 100 else f"   {bd}")

    # Métrique finale
    metric = context.get("final_metric")
    reason = context.get("final_metric_reason")
    if metric:
        print(f"\n📊 Métrique finale: {metric}")
        if reason:
            print(
                f"   Raison: {reason[:80]}..." if len(str(reason)) > 80 else f"   Raison: {reason}"
            )

    # Features enrichies
    features = payload.get("features", [])
    enriched_features = [f for f in features if f.get("feature_description")]

    if enriched_features:
        print(f"\n📝 Features enrichies: {len(enriched_features)}/{len(features)}")
        for feat in enriched_features[:5]:  # Max 5 affichées
            name = feat.get("name", "?")
            desc = feat.get("feature_description", "")
            desc_short = desc[:50] + "..." if len(desc) > 50 else desc
            print(f"   - {name}: {desc_short}")

        if len(enriched_features) > 5:
            print(f"   ... et {len(enriched_features) - 5} autres")

    print("\n" + "=" * 60)
