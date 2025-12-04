from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


def _try_extract_json(raw_string: str) -> dict[str, Any] | None:
    """
    Tente d'extraire un objet JSON d'une string, même si elle contient
    du texte avant/après le JSON (ex: "Voici ma réponse : {...}").

    Returns:
        Le dict parsé ou None si échec.
    """
    # Chercher un bloc JSON entre accolades
    match = re.search(r"\{[\s\S]*\}", raw_string)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def apply_llm_business_annotations(
    snapshot: dict[str, Any],
    llm_result_raw: str | dict[str, Any],
    *,
    min_confidence: float = 0.6,
) -> dict[str, Any]:
    """
    snapshot : ton JSON compact (dict) du dataset
    llm_result_raw : soit un dict déjà parsé, soit une string JSON comme :

        {"Mode":"Final","context":{...},"features":[...]}

    On renvoie un *nouveau* snapshot avec :
      - context.business_description rempli,
      - context.final_metric rempli (métrique choisie par le LLM),
      - context.final_metric_reason rempli (justification métier),
      - features[*].feature_description rempli (par "name"),
    si confidence >= min_confidence.

    Comportement spécial :
    - Si 'context' est absent dans la réponse LLM,
      on cherche les champs à la racine et on affiche un warning.
    - Rétrocompatibilité : si 'metric' existe (ancien format), on l'utilise aussi.
    - Si le JSON est invalide, on retourne le snapshot original sans enrichissement.
    """

    # 0) Parser la sortie LLM si c'est une string JSON
    if isinstance(llm_result_raw, str):
        try:
            llm_result = json.loads(llm_result_raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON invalide du LLM: {e}")
            logger.debug(f"Réponse brute: {llm_result_raw[:500]}...")

            # Tentative d'extraction avec regex
            llm_result = _try_extract_json(llm_result_raw)

            if llm_result is None:
                logger.error(
                    "Impossible de parser la réponse LLM. "
                    "Retour du snapshot original sans enrichissement."
                )
                # On retourne le snapshot original, on ne crash pas
                return deepcopy(snapshot)

            logger.info("JSON extrait avec succès via regex fallback")
    else:
        llm_result = llm_result_raw

    out = deepcopy(snapshot)

    # -----------------------------
    # 1) Contexte global
    # -----------------------------
    ctx_out = out.setdefault("context", {})

    ctx_result = llm_result.get("context")

    # ---- Déterminer la source des données ----
    if isinstance(ctx_result, dict):
        # Cas normal : on lit dans "context"
        src = ctx_result
    else:
        # Fallback : context manquant → on cherche à la racine
        logger.warning(
            "Réponse LLM sans champ 'context' : "
            "tentative de récupération des champs à la racine."
        )
        src = llm_result

    # business_description
    bd = src.get("business_description")
    if isinstance(bd, dict):
        value = bd.get("value")
        conf = bd.get("confidence", 1.0)
        if value and conf >= min_confidence:
            ctx_out["business_description"] = value
            ctx_out.setdefault("_llm_meta", {})["business_description_confidence"] = conf

    # final_metric (nouveau format)
    final_metric = src.get("final_metric")
    if isinstance(final_metric, dict):
        value = final_metric.get("value")
        conf = final_metric.get("confidence", 1.0)
        if value and conf >= min_confidence:
            ctx_out["final_metric"] = value
            ctx_out.setdefault("_llm_meta", {})["final_metric_confidence"] = conf

    # final_metric_reason (justification métier)
    final_metric_reason = src.get("final_metric_reason")
    if isinstance(final_metric_reason, dict):
        value = final_metric_reason.get("value")
        conf = final_metric_reason.get("confidence", 1.0)
        if value and conf >= min_confidence:
            ctx_out["final_metric_reason"] = value
            ctx_out.setdefault("_llm_meta", {})["final_metric_reason_confidence"] = conf

    # metric (ancien format - rétrocompatibilité)
    metric = src.get("metric")
    if isinstance(metric, dict) and "final_metric" not in ctx_out:
        value = metric.get("value")
        conf = metric.get("confidence", 1.0)
        if value and conf >= min_confidence:
            # Stocker dans final_metric pour cohérence
            ctx_out["final_metric"] = value
            ctx_out.setdefault("_llm_meta", {})["final_metric_confidence"] = conf
            logger.info("Ancien format 'metric' converti en 'final_metric'")

    # -----------------------------
    # 2) Features
    # -----------------------------
    features_out = out.setdefault("features", [])
    # index des features par name dans le snapshot
    feat_by_name = {f.get("name"): f for f in features_out if isinstance(f, dict)}

    for f_res in llm_result.get("features", []) or []:
        name = f_res.get("name")
        if not name:
            continue

        fd = f_res.get("feature_description")
        if not isinstance(fd, dict):
            continue

        value = fd.get("value")
        conf = fd.get("confidence", 1.0)

        if not value or conf < min_confidence:
            continue

        # Si la feature existe déjà dans le snapshot, on la met à jour
        if name in feat_by_name:
            feat = feat_by_name[name]
        else:
            # au cas où le LLM décrit une feature non présente
            feat = {"name": name}
            features_out.append(feat)
            feat_by_name[name] = feat

        feat["feature_description"] = value
        meta = feat.setdefault("_llm_meta", {})
        meta["feature_description_confidence"] = conf

    return out
